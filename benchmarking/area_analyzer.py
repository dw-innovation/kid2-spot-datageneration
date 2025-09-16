class AreaAnalyzer:
    """
    Compare a reference geographic area with a generated area and report match quality.

    The expected area schema is a dictionary of the form:
        {
            "type": "bbox" | "area",
            # if type == "bbox": value may be omitted (only the type is compared)
            # if type == "area": value is a string name (e.g., country/region/city)
            "value": <optional, str>
        }

    Notes:
        - Current implementation assumes there is exactly one area to compare
          (i.e., it computes per-singleton metrics and a boolean "perfect" flag).
        - For "area" (name-based) comparison, matching is case-insensitive.
        - Country/name normalization beyond lowercasing is not implemented yet
          (see TODO).
    """
    def __init__(self):
        """Initialize the analyzer. (No state is maintained.)"""
        pass

    def compare_area(self, ref_area, gen_area):
        """
        Compare a reference area against a generated area.

        Args:
            ref_area: Reference area dictionary. Must contain:
                - 'type' (str): Either 'bbox' or 'area'.
                - 'value' (str, optional): If 'type' == 'area', the area name.
            gen_area: Generated area dictionary. Must contain:
                - 'type' (str): Either 'bbox' or 'area'.
                - 'value' (str, optional): If 'type' == 'area', the area name.

        Returns:
            dict: Aggregated comparison metrics for this single area:
                - total_area (int): Always 1 (singleton comparison).
                - total_bbox (int): 1 if ref area type is 'bbox', else 0.
                - total_name_area (int): 1 if ref area type is 'area', else 0.
                - num_correct_bbox (int): 1 if both are 'bbox' (type matches), else 0.
                - num_correct_name_area (int): 1 if both are 'area' and names match
                  case-insensitively, else 0.
                - num_correct_area_type (int): 1 if 'type' matches, else 0.
                - area_perfect_result (bool): True iff all applicable checks pass
                  (type match AND bbox/name match as relevant).

        Raises:
            Exception: If ref_area['type'] is neither 'bbox' nor 'area'.

        Example:
            >> analyzer = AreaAnalyzer()
            >> ref = {"type": "area", "value": "Germany"}
            >> gen = {"type": "area", "value": "germany"}
            >> analyzer.compare_area(ref, gen)["area_perfect_result"]
            True
        """
        num_correct_bbox = 0
        num_correct_name_area = 0
        num_correct_area_type = 0
        perfect_result = False

        # atm we have always 1 area
        total_area = 1

        total_bbox = 0
        total_name_area = 0

        if ref_area['type'] == 'bbox':
            total_bbox+=1
        elif ref_area['type'] == 'area':
            total_name_area += 1
        else:
            raise Exception('Unidentified area type')

        if ref_area['type'] == gen_area['type']:
            num_correct_area_type+=1

            if ref_area['type'] == 'bbox':
                num_correct_bbox+=1
            else:
                if ref_area['value'].lower() == gen_area['value'].lower():
                    num_correct_name_area+=1
                else:
                    # todo: compare country names
                    pass

        if (total_area == (num_correct_name_area+num_correct_bbox)) and \
                (total_bbox==num_correct_bbox) and \
                (total_name_area==num_correct_name_area) and \
                (total_area == num_correct_area_type):
            perfect_result = True

        return dict(
            total_area = total_area,
            total_bbox = total_bbox,
            total_name_area = total_name_area,
            num_correct_bbox = num_correct_bbox,
            num_correct_name_area = num_correct_name_area,
            num_correct_area_type = num_correct_area_type,
            area_perfect_result = perfect_result
        )

    # def compare_areas_strict(self, ref_area, test_area) -> ResultDataType:
    #     """
    #     Check if two areas are identical (strict equality on dicts).
    #
    #     Args:
    #         ref_area: First area dict.
    #         test_area: Second area dict.
    #
    #     Returns:
    #         ResultDataType: TRUE if dicts are exactly equal, else FALSE.
    #     """
    #     return ResultDataType.TRUE if (ref_area == test_area) else ResultDataType.FALSE
    #
    # def compare_areas_light(self, ref_area, test_area) -> ResultDataType:
    #     """
    #     Light comparison that tolerates some formatting differences:
    #     - For named areas, compare lowercased 'value' (or 'name' fallback).
    #     - For bbox, only types must match.
    #
    #     Args:
    #         ref_area: Reference area dict.
    #         test_area: Candidate area dict.
    #
    #     Returns:
    #         ResultDataType: TRUE on light match conditions, else FALSE.
    #     """
    #     if ref_area["type"] != "bbox":
    #         if test_area['type'] == "bbox":
    #             return ResultDataType.FALSE
    #         ref_area['value'] = ref_area['value'].lower()
    #         if 'value' in test_area:
    #             test_area['value'] = test_area['value'].lower()
    #         else:
    #             test_area['value'] = test_area['name'].lower()
    #
    #     else:
    #         # generations sometimes omit the value
    #         # print(ref_area)
    #         # print(test_area)
    #         if ref_area['type'] == test_area['type']:
    #             return ResultDataType.TRUE
    #
    #     # todo: relaxing encoding issue
    #
    #     return self.compare_areas_strict(ref_area=ref_area, test_area=test_area)