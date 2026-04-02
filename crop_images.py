# crop_images.py
# Dictionary of all 22 crops with their image URLs and names
# Images sourced from Wikimedia Commons (free, no API key needed)

CROP_IMAGES = {
    "rice": {
        "name": "Rice",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7b/White_rice.jpg/640px-White_rice.jpg",
        "alt": "Rice crop in paddy field"
    },
    "maize": {
        "name": "Maize",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0a/Corn_aka_Maize.jpg/640px-Corn_aka_Maize.jpg",
        "alt": "Maize corn crop"
    },
    "chickpea": {
        "name": "Chickpea",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/7/72/Chickpeas_Cicer_arietinum.jpg/640px-Chickpeas_Cicer_arietinum.jpg",
        "alt": "Chickpea plant"
    },
    "kidneybeans": {
        "name": "Kidney Beans",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/1/19/RedKidneyBeans.jpg/640px-RedKidneyBeans.jpg",
        "alt": "Kidney beans"
    },
    "pigeonpeas": {
        "name": "Pigeon Peas",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d3/PigeonpeaFlower.jpg/640px-PigeonpeaFlower.jpg",
        "alt": "Pigeon peas plant"
    },
    "mothbeans": {
        "name": "Moth Beans",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4e/Moth_bean_seeds.jpg/640px-Moth_bean_seeds.jpg",
        "alt": "Moth beans"
    },
    "mungbean": {
        "name": "Mung Bean",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/37/Mung_beans.jpg/640px-Mung_beans.jpg",
        "alt": "Mung bean seeds"
    },
    "blackgram": {
        "name": "Black Gram",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c7/Black_gram_seeds.jpg/640px-Black_gram_seeds.jpg",
        "alt": "Black gram seeds"
    },
    "lentil": {
        "name": "Lentil",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6e/Lentil_seeds.jpg/640px-Lentil_seeds.jpg",
        "alt": "Lentil seeds"
    },
    "pomegranate": {
        "name": "Pomegranate",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/7/72/Pomegranate_fruit_-_whole_and_sectioned.jpg/640px-Pomegranate_fruit_-_whole_and_sectioned.jpg",
        "alt": "Pomegranate fruit"
    },
    "banana": {
        "name": "Banana",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8a/Banana-Fruit-Pieces.jpg/640px-Banana-Fruit-Pieces.jpg",
        "alt": "Banana fruit"
    },
    "mango": {
        "name": "Mango",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/9/90/Hapus_Mango.jpg/640px-Hapus_Mango.jpg",
        "alt": "Mango fruit"
    },
    "grapes": {
        "name": "Grapes",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/b/bb/Table_grapes_on_the_vine.jpg/640px-Table_grapes_on_the_vine.jpg",
        "alt": "Grapes on vine"
    },
    "watermelon": {
        "name": "Watermelon",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4c/Watermelon_seedless.jpg/640px-Watermelon_seedless.jpg",
        "alt": "Watermelon fruit"
    },
    "muskmelon": {
        "name": "Muskmelon",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/c/cd/Muskmelon_Fruit.jpg/640px-Muskmelon_Fruit.jpg",
        "alt": "Muskmelon fruit"
    },
    "apple": {
        "name": "Apple",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/1/15/Red_Apple.jpg/640px-Red_Apple.jpg",
        "alt": "Apple fruit"
    },
    "orange": {
        "name": "Orange",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Oranges_and_orange_juice.jpg/640px-Oranges_and_orange_juice.jpg",
        "alt": "Orange fruit"
    },
    "papaya": {
        "name": "Papaya",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Papaya_cross_section_BNC.jpg/640px-Papaya_cross_section_BNC.jpg",
        "alt": "Papaya fruit"
    },
    "coconut": {
        "name": "Coconut",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f2/Coconut_on_white_background.jpg/640px-Coconut_on_white_background.jpg",
        "alt": "Coconut fruit"
    },
    "cotton": {
        "name": "Cotton",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5d/CottonPlant.jpg/640px-CottonPlant.jpg",
        "alt": "Cotton plant"
    },
    "jute": {
        "name": "Jute",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/de/Jute_Field_Bangladesh.jpg/640px-Jute_Field_Bangladesh.jpg",
        "alt": "Jute field"
    },
    "coffee": {
        "name": "Coffee",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/45/A_small_cup_of_coffee.JPG/640px-A_small_cup_of_coffee.JPG",
        "alt": "Coffee beans"
    }
}

def get_crop_image(crop_name):
    """
    Get image URL for a crop.
    Usage: get_crop_image("rice") 
    Returns dict with name, image_url, alt
    Falls back to a default farm image if crop not found.
    """
    key = crop_name.lower().replace(" ", "").replace("-", "")
    return CROP_IMAGES.get(key, {
        "name":      crop_name.capitalize(),
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5f/Farm_and_field.jpg/640px-Farm_and_field.jpg",
        "alt":       f"{crop_name} crop"
    })


# ── CSV Export (optional) ──────────────────────────────────────
if __name__ == "__main__":
    import csv
    rows = []
    for key, val in CROP_IMAGES.items():
        rows.append({
            "crop_key":  key,
            "crop_name": val["name"],
            "image_url": val["image_url"],
            "alt_text":  val["alt"]
        })

    with open("crop_images.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["crop_key", "crop_name", "image_url", "alt_text"])
        writer.writeheader()
        writer.writerows(rows)

    print("✅ crop_images.csv created with all 22 crops")
    print(f"   Total crops: {len(rows)}")
    for r in rows:
        print(f"   {r['crop_name']:15} → {r['image_url'][:60]}...")
