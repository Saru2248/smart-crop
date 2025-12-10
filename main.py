# main.py
from crop_advisory import crop_advisory
from storage_monitor import check_storage
from market_platform import place_order

while True:
    print("\n--- Smart Agriculture System ---")
    print("1. Crop Advisory")
    print("2. Storage Monitoring")
    print("3. Market Order")
    print("4. Exit")

    choice = input("Enter your choice: ")

    if choice == "1":
        crop = input("Enter crop: ")
        soil = input("Enter soil type: ")
        weather = input("Enter weather: ")
        print(crop_advisory(crop, soil, weather))

    elif choice == "2":
        temp = float(input("Enter storage temperature (°C): "))
        humidity = float(input("Enter storage humidity (%): "))
        print(check_storage(temp, humidity))

    elif choice == "3":
        produce = input("Enter produce: ")
        qty = float(input("Enter quantity (kg): "))
        print(place_order(produce, qty))

    elif choice == "4":
        break

    else:
        print("Invalid choice! Try again.")
