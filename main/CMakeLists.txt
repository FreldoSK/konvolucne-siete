idf_component_register(SRCS "main.cpp" "image_processor.cpp"
                       INCLUDE_DIRS ".")

# "idf.py flash" uploads the imgs folder to the ESP flash memory.
littlefs_create_partition_image(storage imgs FLASH_IN_PROJECT)
