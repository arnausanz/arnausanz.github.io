import os

with open("../Data/test.txt", "a") as file:
    """
    Testing automation --> Write the day and time of the execution in a file
    """
    file.write("Day: " + str(os.popen("date +'%d/%m/%Y'").read()).strip() + ' - ' + "Time: " + str(os.popen("date +'%T'").read()).strip())
    file.write("\n")
    file.close()