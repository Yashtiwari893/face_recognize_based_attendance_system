from werkzeug.security import generate_password_hash

password = input("Naya admin password enter karein: ")
hashed_password = generate_password_hash(password)
print("\nAapka hashed password hai:\n")
print(hashed_password)