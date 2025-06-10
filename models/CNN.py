import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.RevIN import RevIN


class Model(nn.Module):
    """
    Convolutional Neural Network (CNN) model with RevIN normalization
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len

        # 预定义输入、隐藏层和输出层的大小
        self.input_channels = configs.enc_in
        self.hidden_channels = configs.d_model
        self.output_channels = configs.c_out

        self.conv1 = nn.Conv1d(in_channels=self.input_channels, out_channels=self.hidden_channels, kernel_size=3,
                               padding=1)
        self.conv2 = nn.Conv1d(in_channels=self.hidden_channels, out_channels=self.hidden_channels, kernel_size=3,
                               padding=1)
        self.conv3 = nn.Conv1d(in_channels=self.hidden_channels, out_channels=self.output_channels, kernel_size=3,
                               padding=1)

        self.revin = RevIN(self.input_channels)

        self.dropout = nn.Dropout(configs.dropout)
        self.act = F.gelu

        if self.task_name == 'classification':
            self.fc_out = nn.Linear(self.output_channels * configs.seq_len, configs.num_class)
        else:
            self.fc_out = nn.Identity()  # 占位层

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        Forward pass for the CNN model.
        Task-specific logic is handled by different branches of the model.
        """
        # 归一化输入
        x_enc = self.revin(x_enc, mode="norm")

        # 调整输入维度以适配Conv1d（[batch_size, channels, seq_len]）
        x = x_enc.permute(0, 2, 1)

        # 第一层卷积
        x = self.conv1(x)
        x = self.act(x)
        x = self.dropout(x)

        # 第二层卷积
        x = self.conv2(x)
        x = self.act(x)
        x = self.dropout(x)

        # 第三层卷积
        x = self.conv3(x)

        # 调整维度以适配任务需求
        x = x.permute(0, 2, 1)  # 将维度还原为 [batch_size, seq_len, channels]

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