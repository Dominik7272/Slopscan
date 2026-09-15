from slopscan import Slopscan


model_path = "./best.pt"
filename = "../test.jpg"

model = Slopscan(model_path)

print(f"{model.classify(filename):.4f}")
