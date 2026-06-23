from parser.profile_parser import extract_patient_profile

profile = extract_patient_profile(
    """
    My name is Sarah.
    I am 28 years old.
    I want to lose weight and improve sleep.
    I sleep about 5 hours per night.
    I struggle with late-night snacking.
    """
)

print(profile.model_dump())