import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.RevIN import RevIN


class Model(nn.Module):
    """
    Multi-Layer Perceptron (MLP) model with RevIN normalization
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len

        # 预定义输入、隐藏层和输出层的大小
        self.input_dim = configs.enc_in
        self.hidden_dim = configs.d_model
        self.output_dim = configs.c_out

        # MLP 层
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, self.output_dim)

        # 使用 RevIN 归一化
        self.revin = RevIN(self.hidden_dim)

        self.dropout = nn.Dropout(configs.dropout)
        self.act = F.gelu  # 激活函数

        if self.task_name == 'classification':
            self.fc_out = nn.Linear(self.hidden_dim * configs.seq_len, configs.num_class)
        else:
            self.fc_out = nn.Identity()

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        batch_size, seq_len, features = x_enc.shape

        # 归一化输入
        x = self.revin(x_enc, mode="norm")

        # 将输入数据展平为 (batch_size * seq_len, features)
        x = x.view(-1, features)

        # 第一层
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)

        # 第二层
        x = self.fc2(x)
        x = self.act(x)
        x = self.dropout(x)

        # 输出层
        x = self.fc3(x)

        # 将输出重新调整为原始的序列形状 (batch_size, seq_len, output_dim)
        x = x.view(batch_size, seq_len, -1)

        # 反归一化
        x = self.revin(x, mode="denorm")

        # 根据任务类型进行不同的处理
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            return x[:, -self.pred_len:, :]

        if self.task_name == 'anomaly_detection':
            return x

        if self.task_name == 'classification':
            output = x.reshape(x.shape[0], -1)  # 展平
            output = self.fc_out(output)  # 分类输出
            return output

        return None
