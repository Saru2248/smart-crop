
def crop_advisory(crop, soil, weather):
    advice = []

    if crop.lower() == "wheat":
        if soil.lower() == "dry":
            advice.append("Irrigate twice a week")
        else:
            advice.append("Irrigate once a week")

        if weather.lower() == "rainy":
            advice.append("Reduce irrigation")
        else:
            advice.append("Monitor soil moisture regularly")

    elif crop.lower() == "rice":
        advice.append("Maintain flooded field conditions")
        advice.append("Check for pest infestation")

    return advice


# Example Usage
crop = "wheat"
soil = "dry"
weather = "sunny"
print("Advice for", crop, ":", crop_advisory(crop, soil, weather))
