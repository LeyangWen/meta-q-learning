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
