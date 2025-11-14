import json

def create_filterable_metadata():
    """
    Reads products.json to extract unique brand and category values,
    and saves them to a JSON file for later use in filtering.
    """
    try:
        with open("products.json", "r", encoding="utf-8") as f:
            products = json.load(f)
    except FileNotFoundError:
        print("Error: products.json not found.")
        return
    except json.JSONDecodeError:
        print("Error: Could not decode JSON from products.json.")
        return

    brands = sorted(list(set(p["brand"] for p in products.values() if p.get("brand"))))
    categories = sorted(list(set(p["category"] for p in products.values() if p.get("category"))))

    filterable_values = {
        "brands": brands,
        "categories": categories,
    }

    with open("filterable_metadata.json", "w", encoding="utf-8") as f:
        json.dump(filterable_values, f, indent=4)

    print("Successfully created filterable_metadata.json")

if __name__ == "__main__":
    create_filterable_metadata()
