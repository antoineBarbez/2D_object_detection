class_names = ["Car", "Van", "Truck", "Pedestrian", "Person_sitting", "Cyclist", "Tram"]


def get_name_to_id_map():
    return {name: i for i, name in enumerate(class_names)}
