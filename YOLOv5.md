1. Replace Focus layer in the yaml or the yolo.py -> parse_model function.
   If do it in parse_model, you can use the pretrained weights.

```
   # For convert to Caffe
   if m is Focus:
         args.append(2)
         m = eval("Conv")
```

2. Set the MaxPooling ceil_mode to True. Common.py -> SPP layer.

```
   class SPP(nn.Module):
      # Spatial pyramid pooling layer used in YOLOv3-SPP
      def __init__(self, c1, c2, k=(5, 9, 13)):
         super().__init__()
         c_ = c1 // 2  # hidden channels
         self.cv1 = Conv(c1, c_, 1, 1)
         self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
         self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2, ceil_mode=True) for x in k])
```

3. Add an export function on Detect layer for caffe convert, according to your target device post_process function.

As an example:

```
   def forward_export(self, x):
      # x = x.copy()  # for profiling
      z = []  # inference output
      for i in range(self.nl):
         x[i] = self.m[i](x[i])  # conv

      return x
```

4. Retrain your model and convert to caffe. 

Replace the forward function as forward_export(as the example above).

Good luck!