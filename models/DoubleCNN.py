import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.RevIN import RevIN

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.dCNN_mode = configs.dCNN_mode
        self.use_norm = configs.use_norm
        self.dCNN_use_norm = configs.dCNN_use_norm

        if self.use_norm:
            self.revin = RevIN(configs.enc_in)
        self.dCNN_layer = Double2DCNNLayer(enc_in=self.enc_in, seq_len=self.seq_len,
                                           mode=self.dCNN_mode, use_norm=self.dCNN_use_norm)
        self.fc = nn.Linear(self.seq_len * self.enc_in, self.pred_len * self.enc_in)
        self.final_activation = nn.GELU()

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        x_enc: batch * seq_len * enc_in
        """
        if self.use_norm:
            x_enc = self.revin(x_enc.permute(0, 2, 1)).permute(0, 2, 1)

        out = self.dCNN_layer(x_enc)

        if self.use_norm:
            out = self.revin(out.permute(0, 2, 1), mode='denorm').permute(0, 2, 1)
        else:
            out = out

        out = out.view(out.shape[0], -1)  # 转换为 (batch_size, seq_len * enc_in)
        out = self.fc(out)  # 转换为 (batch_size, pred_len * enc_in)
        out = self.final_activation(out)
        out = out.view(out.shape[0], self.pred_len, self.enc_in)  # 转换为 (batch_size, pred_len, enc_in)

        return out


class DoubleCNNLayer(nn.Module):
    """
    DoubleCNN.py layer for time series forecasting.
    This layer uses two CNN layers to capture time dependencies and variable dependencies respectively.
    """

    def __init__(self, enc_in, seq_len, out_len,
                 time_kernel_size, variable_kernel_size):
        super(DoubleCNNLayer, self).__init__()
        self.enc_in = enc_in
        self.seq_len = seq_len
        self.out_len = out_len

        # Time dimension CNN
        self.time_cnn = nn.Conv1d(in_channels=self.enc_in,
                                  out_channels=self.enc_in,
                                  kernel_size=time_kernel_size,
                                  padding=time_kernel_size//2,
                                  stride=1)
        self.time_cnn_GELU = nn.GELU()

        # Variable dimension CNN
        self.variable_cnn = nn.Conv1d(in_channels=self.seq_len,
                                      out_channels=self.seq_len,
                                      kernel_size=variable_kernel_size,
                                      padding=variable_kernel_size//2,
                                      stride=1)
        self.variable_cnn_GELU = nn.GELU()

        self.mlp = nn.Sequential(nn.Linear(self.seq_len * self.enc_in * 2, 512),
                                 nn.GELU(),
                                 nn.Linear(512, 256),
                                 nn.GELU(),
                                 nn.Linear(256, self.out_len * self.enc_in))

    def forward(self, x_enc):
        """
        x_enc: batch * seq_len * enc_in
        """
        x_enc = x_enc.permute(0, 2, 1)

        # Time dimension convolution
        time_cnn_out = self.time_cnn_GELU(self.time_cnn(x_enc))

        # Variable dimension convolution
        variable_cnn_out = self.variable_cnn_GELU(self.variable_cnn(x_enc.permute(0, 2, 1)))

        combined_out = torch.cat((variable_cnn_out.view(variable_cnn_out.size(0), -1), time_cnn_out.view(time_cnn_out.size(0), -1)), dim=-1)
        out = self.mlp(combined_out)
        out = out.view(out.size(0), self.out_len, self.enc_in)

        return out

class Double2DCNNLayer(nn.Module):
    def __init__(self, enc_in, seq_len, mode, use_norm=True):
        super(Double2DCNNLayer, self).__init__()
        self.mode = mode
        self.enc_in = enc_in
        self.seq_len = seq_len
        self.use_norm = use_norm

        self.time_cnn = nn.Conv2d(in_channels=1,
                                  out_channels=1,
                                  kernel_size=(enc_in, 1),
                                  padding=(0, 0),
                                  stride=1)
        self.time_cnn_activation = nn.GELU()

        self.variable_cnn = nn.Conv2d(in_channels=1,
                                      out_channels=1,
                                      kernel_size=(1, seq_len),
                                      padding=(0, 0),
                                      stride=1)
        self.variable_cnn_activation = nn.GELU()

        if self.use_norm:
            self.time_norm = nn.LayerNorm([1, 1, seq_len])
            self.variable_norm = nn.LayerNorm([1, enc_in, 1])
            self.final_norm = nn.LayerNorm([seq_len, enc_in])

    def forward(self, x_enc):
        """
        x_enc: batch * seq_len * enc_in
        """
        # Reshape input to (batch_size, 1, enc_in, seq_len)
        x_enc = x_enc.permute(0, 2, 1).unsqueeze(1)

        if self.mode == 'res':
            time_cnn_out = self.time_cnn(x_enc)  # Output shape: batch_size * 1 * 1 * seq_len
            if self.use_norm:
                time_cnn_out = self.time_norm(time_cnn_out)
            time_cnn_out = self.time_cnn_activation(time_cnn_out)

            variable_cnn_out = self.variable_cnn(x_enc)  # Output shape: batch_size * 1 * enc_in * 1
            if self.use_norm:
                variable_cnn_out = self.variable_norm(variable_cnn_out)
            variable_cnn_out = self.variable_cnn_activation(variable_cnn_out)

            # 处理时间卷积输出：扩展变量维度
            time_cnn_out_expanded = time_cnn_out.expand(-1, -1, self.enc_in , -1).contiguous()
            # 处理变量卷积输出：扩展时间维度
            variable_cnn_out_expanded = variable_cnn_out.expand(-1, -1, -1, self.seq_len).contiguous()

            # Residual connection
            residual = x_enc
            combined = residual + time_cnn_out_expanded + variable_cnn_out_expanded

            out = combined.squeeze(1).permute(0, 2, 1)
            if self.use_norm:
                out = self.final_norm(out)

        elif self.mode == 'weight':
            time_weights = self.time_cnn(x_enc)
            if self.use_norm:
                time_weights = self.time_norm(time_weights)
            time_weights = self.time_cnn_activation(time_weights)

            variable_weights = self.variable_cnn(x_enc)
            if self.use_norm:
                variable_weights = self.variable_norm(variable_weights)
            variable_weights = self.variable_cnn_activation(variable_weights)

            time_weights_expanded = time_weights.expand(-1, -1, self.enc_in, -1).contiguous()
            variable_weights_expanded = variable_weights.expand(-1, -1, -1, self.seq_len).contiguous()

            combined_weights = time_weights_expanded * variable_weights_expanded
            weighted_data = x_enc * combined_weights

            out = weighted_data.squeeze(1).permute(0, 2, 1)
            if self.use_norm:
                out = self.final_norm(out)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}. Expected 'res' or 'weight'.")
        return out