import csv
import os

def extract_columns_to_files(input_file, output_dir):
    with open(input_file, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        columns = [col for col in reader.fieldnames if
                   reader.fieldnames.index(col) >= reader.fieldnames.index('poisson')]

        for col in columns:
            output_file = os.path.join(output_dir, f"{col}.txt")
            with open(output_file, 'w') as txtfile:
                for row in reader:
                    value = format(float(row[col]), '.4f')
                    txtfile.write(value + '\n')
                # Reset the CSV reader's position to the start of the file for the next column
                csvfile.seek(0)
                next(reader)


# extract_columns_to_files(input_file= 'data/debatepedia/quad_debatepedia/Debatepedia_arguments.csv', output_dir='data/debatepedia/quad_debatepedia/features')

extract_columns_to_files(input_file= 'data/angrymen/quad-AngryMen/TwelveAngryMan_arguments.csv', output_dir='data/angrymen/quad-AngryMen/features')