import sys
from tqdm import tqdm

if __name__ == "__main__":
    input_filename = sys.argv[1]
    output_filename = sys.argv[2]
    size = int(sys.argv[3])
    with open(input_filename) as input_file:
        with open(output_filename, "w") as output_file:
            for i, line in enumerate(tqdm(input_file)):
                if i == 0:
                    output_file.write(line.strip() + "\n")
                    continue
                values = line.strip().split()
                if int(float(values[2])) == size:
                    output_file.write(line.strip() + "\n")
