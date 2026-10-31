import torch

def inspect(name, t):
    print(f"\n{name}")
    print("shape:", t.shape)
    print("stride:", t.stride())
    print("contiguous:", t.is_contiguous())
    print("nbytes:", t.numel() * t.element_size())
    print("data_ptr:", t.data_ptr())

x = torch.arange(12).reshape(3, 4)
y = x.transpose(0, 1)
z = y.contiguous()
r = y.reshape(12)

for name, t in [
    ("x", x),
    ("y", y),
    ("z", z),
    ("r", r),
]:
    inspect(name, t)

print("\nStorage sharing:")
print("x/y:", x.untyped_storage().data_ptr()
      == y.untyped_storage().data_ptr())
print("x/z:", x.untyped_storage().data_ptr()
      == z.untyped_storage().data_ptr())
print("x/r:", x.untyped_storage().data_ptr()
      == r.untyped_storage().data_ptr())

try:
    v = y.view(12)
except RuntimeError as e:
    print("\nview failed:", e)