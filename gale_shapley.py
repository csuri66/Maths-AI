def solve(men_preferences, women_preferences):
    n = len(men_preferences)
    free_men = list(men_preferences.keys())  # All men start out free
    engaged_pairs = {}  # To store engaged pairs
    men_proposals = {man: [] for man in men_preferences}  # Track proposals

    # While there are free men who haven't proposed to all women
    while free_men:
        man = free_men[0]  # Grab the first free man
        man_pref = men_preferences[man]
        for woman in man_pref:
            if woman not in men_proposals[man]:
                men_proposals[man].append(woman)  # Record the proposal

                # If the woman is free, engage them
                if woman not in engaged_pairs.values():
                    engaged_pairs[man] = woman
                    free_men.remove(man)
                    break
                else:
                    # If the woman is already engaged, check if she prefers the new man
                    current_man = next(m for m, w in engaged_pairs.items() if w == woman)
                    woman_pref = women_preferences[woman]
                    if woman_pref.index(man) < woman_pref.index(current_man):
                        # Engage the new man and free the current one
                        engaged_pairs[man] = woman
                        free_men.remove(man)
                        free_men.append(current_man)
                        del engaged_pairs[current_man]
                        break

    return engaged_pairs


