"""Classify matched and unmatched building instances."""

from __future__ import annotations


def classify_changes(before, after, matching, area_change_threshold=0.30, changed_mask_iou_threshold=0.50):
    changes = []
    for match in matching["matches"]:
        first = before["instances"][match["before_index"]]
        second = after["instances"][match["after_index"]]
        before_area = int(first["mask"].sum())
        after_area = int(second["mask"].sum())
        area_change = after_area - before_area
        area_percent = area_change / before_area if before_area else 0.0
        changed = abs(area_percent) >= area_change_threshold or match["mask_iou"] < changed_mask_iou_threshold
        changes.append({
            "before_id": first["instance_id"], "after_id": second["instance_id"],
            "classification": "CHANGED_BUILDING" if changed else "UNCHANGED",
            "before_area": before_area, "after_area": after_area,
            "area_change": area_change, "area_change_percent": area_percent * 100.0,
            "mask_iou": match["mask_iou"], "bounding_box_iou": match["box_iou"],
            "centroid_distance": match["centroid_distance"],
            "before_score": first["score"], "after_score": second["score"],
        })
    for index in matching["unmatched_before"]:
        first = before["instances"][index]
        changes.append({"before_id": first["instance_id"], "after_id": None, "classification": "REMOVED_BUILDING", "before_area": int(first["mask"].sum()), "after_area": 0, "area_change": -int(first["mask"].sum()), "area_change_percent": -100.0, "mask_iou": 0.0, "bounding_box_iou": 0.0, "centroid_distance": None, "before_score": first["score"], "after_score": None})
    for index in matching["unmatched_after"]:
        second = after["instances"][index]
        changes.append({"before_id": None, "after_id": second["instance_id"], "classification": "NEW_BUILDING", "before_area": 0, "after_area": int(second["mask"].sum()), "area_change": int(second["mask"].sum()), "area_change_percent": None, "mask_iou": 0.0, "bounding_box_iou": 0.0, "centroid_distance": None, "before_score": None, "after_score": second["score"]})
    return changes