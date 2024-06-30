from __future__ import annotations  # for my python 3.8 env
from utility.DataBuffer import *
from deprecated import deprecated
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class CriteriaChecker:
    """
    Class used to determine if certain criteria are met
    """

    @staticmethod
    def satisfy_valence(valence) -> bool:
        """
        Static Method. 
        Check if the valence satisfies the criteria
        --return: Bool indicating if satisfied
        """
        return valence > 0

    @staticmethod
    def satisfy_arousal(arousal) -> bool:
        """
        Static Method.
        Check if the arousal satisfies the criteria
        --return: Bool indicating if satisfied
        """
        return arousal > 0

    @staticmethod
    def satisfy_engagement(engagement, normalized=True, eng_centroids=None, eng_normalized_centroids=None) -> bool:
        """
        Static Method.
        Check if the engagement satisfies the requirement
        --return: Bool indicating if satisfied
        """

        # Check if the response is normalized and using it accordingly
        if normalized:
            engagement_centroids = eng_normalized_centroids
        else:
            engagement_centroids = eng_centroids

        # Calculate distance from the point and the centroid
        # Find if it's closest to the middle one
        engagement_distances = np.abs(engagement_centroids - engagement)
        engagement_closest = np.argmin(engagement_distances)

        return engagement_closest == 1

    @staticmethod
    def satisfy_vigilance(vigilance, normalized=True, vig_centroids=None, vig_normalized_centroids=None) -> bool:
        """
        Static Method.
        Check if the vigilance satisfies the requirement
        --return: Bool indicating if satisfied
        """

        # Check if the response is normalized and using it accordingly
        if normalized:
            vigilance_centroids = vig_normalized_centroids
        else:
            vigilance_centroids = vig_centroids

        # Calculate distance from the point and the centroid
        # Find if it's closest to the middle one
        vigilance_distances = np.abs(vigilance_centroids - vigilance)
        vigilance_closest = np.argmin(vigilance_distances)

        return vigilance_closest == 1

    @staticmethod
    def satisfy_check(human_response, normalized=True,
                      eng_centroids=None, vig_centroids=None,
                      eng_normalized_centroids=None, vig_normalized_centroids=None) -> tuple[int, str]:
        """
        Method that will check if each human response satisfies the critiera first
        Return: Number of satisfied, and stirng of satisfy type
        """

        current_satisfy_type = ""
        current_satisfy_number = 0

        # Index Map for Human Response
        index_map = {
            0: "VAL",
            1: "ARO",
            2: "ENG",
            3: "VIG"
        }

        # Criteria Check Method Map for Human Response
        methods_map = {
            "VAL": CriteriaChecker.satisfy_valence,
            "ARO": CriteriaChecker.satisfy_arousal,
            "ENG": CriteriaChecker.satisfy_engagement,
            "VIG": CriteriaChecker.satisfy_vigilance
        }

        # Check for each response
        for i in range(len(human_response)):
            is_satisfy_type = False
            response_value = human_response[i]
            response_type = index_map[i]
            check_method = methods_map[response_type]

            # Check if it is engagement
            if response_type == "ENG":
                is_satisfy_type = check_method(response_value, normalized=normalized,
                                               eng_centroids=eng_centroids, eng_normalized_centroids=eng_normalized_centroids)
            elif response_type == "VIG":
                is_satisfy_type = check_method(response_value, normalized=normalized,
                                               vig_centroids=vig_centroids, vig_normalized_centroids=vig_normalized_centroids)
            else:
                is_satisfy_type = check_method(response_value)

            # Check if this response satisfies the criteria
            if is_satisfy_type:
                current_satisfy_number += 1
                current_satisfy_type += response_type
                # Use '-' to connect
                current_satisfy_type += "-" if i < len(
                    human_response) - 1 else ""

        # Return result
        return current_satisfy_number, current_satisfy_type

    @deprecated
    def satisfy_engagement_vigilance(human_response, normalized=False, eng_centroids=None, vig_centroids=None, eng_normalized_centroids=None, vig_normalized_centroids=None) -> bool:
        """
        Static Method. 
        Check if the engagement and vigilance satisfy the requirement
        --return: Bool indicating if satisfied
        """

        engagement = human_response[2]
        vigilance = human_response[3]

        # Check if the response is normalized. Select accordingly
        if normalized:
            engagement_centroids = eng_normalized_centroids
            vigilance_centroids = vig_normalized_centroids
        else:
            engagement_centroids = eng_centroids
            vigilance_centroids = vig_centroids

        # Calculate distance from the point and the centroid
        # Find if it's closest to the middle one
        engagement_distances = np.abs(engagement_centroids - engagement)
        vigilance_distances = np.abs(vigilance_centroids - vigilance)

        engagement_closest = np.argmin(engagement_distances)
        vigilance_closest = np.argmin(vigilance_distances)

        return engagement_closest == 1 and vigilance_closest == 1

    @deprecated
    def satisfy_all_requirements(human_response, normalized=False, eng_centroids=None, vig_centroids=None, eng_normalized_centroids=None, vig_normalized_centroids=None) -> tuple[bool, bool]:
        """
        Static Method. 
        Check if valence_arousal satisfied and engagement_vigilance satisfied
        --return: 2 bools: [valence_arousal_satisfied, engagement_vigilance_satisifed]
        """
        return CriteriaChecker.satisfy_valence_arousal(human_response), CriteriaChecker.satisfy_engagement_vigilance(
            human_response, normalized=normalized, eng_centroids=eng_centroids, vig_centroids=vig_centroids,
            eng_normalized_centroids=eng_normalized_centroids, vig_normalized_centroids=vig_normalized_centroids)
