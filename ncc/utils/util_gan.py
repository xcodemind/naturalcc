# ref: https://github.com/eric-xw/AREL

class AlterFlag:
    def __init__(self, D_iters: int, G_iters: int, always=None):
        self.D_iters = D_iters
        self.G_iters = G_iters

        self.flag = 'disc'
        self.iters = self.D_iters
        self.curr = 0
        self.always = always

    def inc(self):
        self.curr += 1
        if self.curr >= self.iters and self.always is None:
            if self.flag == 'disc':
                self.flag = 'gen'
                self.iters = self.G_iters
            elif self.flag == 'gen':
                self.flag = 'disc'
                self.iters = self.D_iters
            self.curr = 0