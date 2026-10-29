
# Find-S Algorithm
# BAI702 - Machine Learning II
#
# Goal: Learn the most specific hypothesis that covers
# all positive training examples.

# Each example has four attributes:
# Sky, Temperature, Humidity, Wind

training_data = [
    ["Sunny", "Warm", "Normal", "Strong"],
    ["Sunny", "Warm", "High", "Strong"],
    ["Rainy", "Cold", "High", "Weak"],
    ["Sunny", "Warm", "Normal", "Weak"],
]

# "+" means positive example; "-" means negative example.
labels = ["+", "+", "-", "+"]


def find_s(examples, labels):
    hypothesis = None

    for example, label in zip(examples, labels):
        # Find-S learns only from positive examples.
        if label != "+":
            continue

        # First positive example initializes the hypothesis.
        if hypothesis is None:
            hypothesis = example.copy()
        else:
            # Generalize any attribute that differs.
            for i in range(len(hypothesis)):
                if hypothesis[i] != example[i]:
                    hypothesis[i] = "?"

        print("Positive example:", example)
        print("Current hypothesis:", hypothesis)

    return hypothesis


print("===== FIND-S ALGORITHM =====")
final_hypothesis = find_s(training_data, labels)

print("\nFinal hypothesis:", final_hypothesis)
print("'?' means any value is accepted for that attribute.")
