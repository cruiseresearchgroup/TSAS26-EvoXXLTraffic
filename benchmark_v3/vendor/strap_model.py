import torch
from torch import nn
from torch.nn import functional as F
from .gcn_conv import BatchGCNConv
from .STRAP import STRAP

class STGNN_Backbone(nn.Module):
    def __init__(self, args):
        super(STGNN_Backbone, self).__init__()
       
        self.gcn1 = BatchGCNConv(args.gcn["in_channel"], args.gcn["hidden_channel"], bias=True, gcn=False)
        self.gcn2 = BatchGCNConv(args.gcn["hidden_channel"], args.gcn["out_channel"], bias=True, gcn=False)
        
        self.tcn = nn.Conv1d(
            in_channels=args.tcn["in_channel"], 
            out_channels=args.tcn["out_channel"], 
            kernel_size=args.tcn["kernel_size"],
            dilation=args.tcn["dilation"], 
            padding=int((args.tcn["kernel_size"]-1)*args.tcn["dilation"]/2)
        )
        
    def forward(self, x, adj):
    
        x = torch.nn.functional.relu(self.gcn1(x, adj))  # [bs, N, hidden]
        
        B, N, H = x.shape
        
     
        x = x.reshape(B*N, 1, H)  # [bs*N, 1, hidden]
        x = self.tcn(x)  # [bs*N, 1, hidden]
        
    
        x = x.reshape(B, N, H)  # [bs, N, hidden]
        x = self.gcn2(x, adj)  # [bs, N, out]
        
        return x

BACKBONES = {"stgnn": STGNN_Backbone}

class RAP_Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.dropout = args.dropout
        self.backbone = BACKBONES[getattr(args, "backbone_type", "stgnn")](args)
        self.fc = nn.Linear(args.gcn["out_channel"], args.y_len)
        self.activation = nn.GELU()
        self.use_strap = getattr(args, "use_strap", True)
        self.current_year = args.year
        if self.use_strap:
            self.strap = STRAP(args)
            self.strap_adapter = nn.Linear(args.gcn["out_channel"], self.strap.feature_dim)

    @property
    def pattern_initialized(self):
        return self.use_strap and self.strap.current_year == self.current_year

    def count_parameters(self):
        self.args.logger.info("Total Parameters: %d", sum(p.numel() for p in self.parameters()))
        self.args.logger.info(
            "Trainable Parameters: %d", sum(p.numel() for p in self.parameters() if p.requires_grad)
        )

    def initialize_patterns(self, data, adj, force=False):
        if not self.use_strap:
            return False
        if force or not self.strap.switch_to_year(self.current_year):
            was_training = self.backbone.training
            self.backbone.eval()
            self.strap.extract_patterns(data, adj, self.current_year, self.backbone)
            self.backbone.train(was_training)
        self.args.logger.info("STRAP year %s pattern sources: %s", self.current_year, self.strap.source_counts)
        return True

    def update_patterns(self, data, adj, year=None):
        if year is not None:
            self.current_year = year
        return self.initialize_patterns(data, adj, force=True)

    def set_year(self, year):
        self.current_year = year
        return self.use_strap and self.strap.switch_to_year(year)

    def feature(self, data, adj):
        if self.use_strap and not self.pattern_initialized:
            raise RuntimeError(f"Initialize the STRAP training library for {self.current_year} before prediction")
        x = data.x.reshape(-1, adj.shape[0], self.args.gcn["in_channel"])
        features = self.backbone(x, adj).reshape(-1, self.args.gcn["out_channel"])
        if self.use_strap:
            features = self.strap(self.strap_adapter(features))
            features = F.adaptive_avg_pool1d(
                features.unsqueeze(1), self.args.gcn["out_channel"]
            ).squeeze(1)
        return features + data.x

    def forward(self, data, adj):
        x = self.fc(self.activation(self.feature(data, adj)))
        return F.dropout(x, p=self.dropout, training=self.training)
