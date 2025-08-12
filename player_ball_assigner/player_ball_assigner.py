import sys 
sys.path.append('../')
from utils import get_center_of_bbox, measure_distance

class PlayerBallAssigner():
    def __init__(self):
        self.max_player_ball_distance = 70
    
    def assign_ball_to_player(self,players,ball_bbox):
        ball_position = get_center_of_bbox(ball_bbox)

        miniumum_distance = 99999
        assigned_player=-1

        for player_id, player in players.items():
            player_bbox = player['bbox']

            distance_left = measure_distance((player_bbox[0],player_bbox[-1]),ball_position)
            distance_right = measure_distance((player_bbox[2],player_bbox[-1]),ball_position)
            distance = min(distance_left,distance_right)

            if distance < self.max_player_ball_distance:
                if distance < miniumum_distance:
                    miniumum_distance = distance
                    assigned_player = player_id

        return assigned_player

    def assign_ball_to_player_field(self, players, ball_position_transformed, threshold_meters: float = 2.0):
        """Assign ball to the nearest player using field-scale coordinates (meters).

        Parameters
        - players: dict mapping player_id -> track info dict; expects key 'position_transformed'
        - ball_position_transformed: tuple/list (x, y) in meters; if None, no assignment is made
        - threshold_meters: gating radius in meters for possession

        Returns
        - assigned_player_id (int) or -1 if none within threshold
        """
        if ball_position_transformed is None:
            return -1

        minimum_distance = float("inf")
        assigned_player = -1

        for player_id, player in players.items():
            player_pos_transformed = player.get('position_transformed')
            if player_pos_transformed is None:
                continue

            distance_meters = measure_distance(player_pos_transformed, ball_position_transformed)
            if distance_meters < threshold_meters and distance_meters < minimum_distance:
                minimum_distance = distance_meters
                assigned_player = player_id

        return assigned_player