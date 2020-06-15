import json

test = open("test.json", 'r')
alternates = open("alternates.json", 'a')

for i in range(45):
    test.readline()

for i in range(5):
    line = test.readline()
    code = json.loads(line)["code"]
    print(code)
    print("")
    print(json.loads(line)["nl"])
    newnl = input("Enter Alternate: ")
    if newnl != "":
        alternate = {"code":code, "nl":newnl}
        json.dump(alternate, alternates)
        alternates.write("\n")