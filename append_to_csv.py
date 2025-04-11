import csv

# Function to append a new row to the CSV file
def append_to_csv(file_path, url, gender):
    if gender.lower() not in ['male', 'female', 'both']:
        print("Invalid gender. Please use 'male', 'female', or 'both'.")
        return
    if gender == 'both':
        description = 'Man and woman in a lab'
    elif gender == 'male':
        description = 'Man in a lab'
    else:
        description = 'Woman in a lab'

    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([url, description, gender])
        print(f"Row appended: {url}, {description}, {gender}")

while True:
    url = input("Enter URL (or type 'exit' to quit): ")
    if url.lower() == 'exit':
        break
    try:
        gender_input = int(input("Enter gender (0 for male, 1 for female, 2 for both): "))
        if gender_input not in [0, 1, 2]:
            print("Invalid input. Please enter 0, 1, or 2.")
            continue
        gender = ['male', 'female', 'both'][gender_input]
        append_to_csv('/home/shardul.junagade/my-work/domain-adaptation-brick-kilns/Bias-in-Vision-Pipelines/data.csv', url, gender)
    except ValueError:
        print("Invalid input. Please enter a number (0, 1, or 2).")