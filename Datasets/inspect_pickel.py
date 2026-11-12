
import pickletools

with open("footIQ_best_model_v5.pkl", "rb") as file:
    data = file.read()

print("--- MODEL FILE SIZE ---")
print(len(data), "bytes")

print("\n--- IMPORTANT PICKLE REFERENCES ---")

for opcode, argument, position in pickletools.genops(data):
    if opcode.name in ["GLOBAL", "STACK_GLOBAL"]:
        print(opcode.name, argument, "at position", position)
    elif isinstance(argument, str):
        if any(word in argument.lower() for word in ["pipeline", "dtype", "sklearn", "numpy"]):
            print(opcode.name, repr(argument), "at position", position)
