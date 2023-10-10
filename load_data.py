from cancer_classification.utils import (  # load_csv,; load_file,; load_image,
    load_thumbnails_data,
)


train_path = load_thumbnails_data()
print(train_path)

######
# or #
######

# image_idx = 1020
# load_image(image_idx)


# filename = "train.csv"
# load_file(filename)

# # or

# load_csv() # скачает все .csv
