from itertools import product

FPS_VALUES = [5, 10, 25, 50]

all_schedules = list(product(FPS_VALUES, repeat=4))

with open("height_schedules.txt", "w") as f:
    f.write("HEIGHT_SCHEDULES = {\n")

    for i, schedule in enumerate(all_schedules, start=1):
        f.write(f'    "S{i:03d}": {list(schedule)},\n')

    f.write("}\n")

print(f"Generated {len(all_schedules)} schedules.")
print("Saved to: height_schedules.txt")