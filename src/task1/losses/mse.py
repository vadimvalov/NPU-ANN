class MSELoss:
    def forward(self, pred, target):
        self.pred = pred
        self.target = target
        return ((pred - target) ** 2).mean()

    def backward(self):
        return 2 * (self.pred - self.target) / self.target.size