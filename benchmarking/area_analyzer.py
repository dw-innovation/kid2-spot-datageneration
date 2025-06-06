class AreaAnalyzer:
    def __init__(self):
        pass

    def compare_area(self, ref_area, gen_area):
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
    #     Checks if two areas are identical.
    #
    #     :param area1: The first area to compare.
    #     :param area2: The second area to compare.
    #     :return: Boolean whether the two areas are the same.
    #     """
    #     return ResultDataType.TRUE if (ref_area == test_area) else ResultDataType.FALSE
    #
    # def compare_areas_light(self, ref_area, test_area) -> ResultDataType:
    #     """
    #     Checks if two areas are identical.
    #
    #     :param area1: The first area to compare.
    #     :param area2: The second area to compare.
    #     :return: Boolean whether the two areas are the same.
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