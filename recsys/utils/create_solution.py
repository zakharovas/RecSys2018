import json
import sys

if __name__ == "__main__":
    recommendations_filename = sys.argv[1]
    track_names_filename = sys.argv[2]
    output_filename = sys.argv[3]
    with open(track_names_filename) as track_file:
        names = json.load(track_file)
        decoding = [""] * len(names)
        for name in names:
            decoding[names[name]] = name
    with open(recommendations_filename) as recommendations_file, open(output_filename, "w") as output_file:
        output_file.write("team_info,main,MIPT_MSU,aleksandr.zakharov@phystech.edu\n")
        for line in recommendations_file:
            result = json.loads(line)
            output_file.write("{}, ".format(result["pid"]))
            output_file.write(", ".join([decoding[x] for x in result["recommended"][:500]]) + "\n")
