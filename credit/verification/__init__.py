
from credit.verification.goes_10km_verification import verification as goes_10km_verification
valid_verification = {
    "goes-10km": goes_10km_verification
}



def load_verification(conf):
    try:
        return valid_verification[conf["verification_type"]]
    except KeyError:
        raise KeyError(f"{conf['verification_type']} is not a valid verification type")
