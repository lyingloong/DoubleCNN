import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.RevIN import RevIN


class Model(nn.Module):
    """
    Recurrent Neural Network (RNN) model with RevIN normalization
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len

        # 预定义输入、隐藏层和输出层的大小
        self.input_dim = configs.enc_in
        self.hidden_dim = configs.d_model
        self.output_dim = configs.c_out
        if hasattr(configs, 'num_layers'):
            self.num_layers = configs.num_layers
        else:
            self.num_layers = 3

        self.revin = RevIN(self.input_dim)

        self.rnn = nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers,
                           batch_first=True, dropout=configs.dropout if self.num_layers > 1 else 0)
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)

        if self.task_name == 'classification':
            self.fc_out = nn.Linear(self.output_dim * configs.seq_len, configs.num_class)
        else:
            self.fc_out = nn.Identity()  # 占位层

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        Forward pass for the RNN model.
        Task-specific logic is handled by different branches of the model.
        """
        # 归一化输入
        x_enc = self.revin(x_enc, mode="norm")

        # 通过 RNN 计算
        x, _ = self.rnn(x_enc)
        x = self.fc(x)

        # 反归一化
        x = self.revin(x, mode="denorm")

        # 根据任务类型进行不同的处理
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            return x[:, -self.pred_len:, :]  # 只返回预测部分（预测长度）

        if self.task_name == 'anomaly_detection':
            return x  # 对于异常检测，可以直接返回模型的输出作为分数

        if self.task_name == 'classification':
            # 分类任务时需要 reshaping
            output = x.reshape(x.shape[0], -1)  # 展平
            output = self.fc_out(output)  # 分类输出
            return output

        return None
