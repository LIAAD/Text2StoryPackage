"""
Temporal Inference Engine for DRS Enrichment

This module implements linguistic rules for inferring temporal relations
based on event types (STATE, PROCESS, TRANSITION), tense, aspect, and verb form.

Theoretical foundations:
- Allen's Interval Algebra for temporal relations
- Reichenbach's temporal framework (S, E, R)
- Aspectual properties (Vendler, Smith)
- Discourse structure principles
"""

from nltk.sem.drt import DrtExpression
from collections import defaultdict
from itertools import combinations

dexpr = DrtExpression.fromstring


class TemporalInferenceEngine:
    """
    Inference engine that applies linguistic rules to enrich DRS with temporal relations.
    """

    def __init__(self):
        self.rules = {
            'tense_ordering': self.rule_tense_ordering,
            'aspect_duration': self.rule_aspect_duration,
            'event_type_temporal': self.rule_event_type_temporal,
            'transition_consequence': self.rule_transition_consequence,
            'state_overlap': self.rule_state_overlap,
            'process_duration': self.rule_process_duration,
            'perfective_completion': self.rule_perfective_completion,
            'participle_result_state': self.rule_participle_result_state,
            'temporal_anchor': self.rule_temporal_anchor,
            'narrative_progression': self.rule_narrative_progression,
            'causal_temporal': self.rule_causal_temporal,
        }

        self.inferred_relations = []

    def extract_event_attributes(self, ann_dict):
        """
        Extract all relevant attributes for each event from BRAT annotations.

        Returns:
            dict: {event_id: {'type': ..., 'tense': ..., 'aspect': ..., 'vform': ..., 'pos': ...}}
        """
        events_attrs = {}

        # First pass: get basic event info
        for ann in ann_dict:
            if ann['ann_type'] == 'TextBound' and ann.get('attribute') == 'Event':
                event_id = ann['tag_id']
                events_attrs[event_id] = {
                    'tag_id': event_id,
                    'value': ann['value'],
                    'type': None,
                    'tense': None,
                    'aspect': None,
                    'vform': None,
                    'pos': None
                }

        # Second pass: add attributes
        for ann in ann_dict:
            if ann['ann_type'] == 'Attribute' and ann['tag_ref'] in events_attrs:
                attr_type = ann['attr_type']
                value = ann['value']

                if attr_type == 'Type':
                    events_attrs[ann['tag_ref']]['type'] = value
                elif attr_type == 'Tense':
                    events_attrs[ann['tag_ref']]['tense'] = value
                elif attr_type == 'Aspect':
                    events_attrs[ann['tag_ref']]['aspect'] = value
                elif attr_type == 'Vform':
                    events_attrs[ann['tag_ref']]['vform'] = value
                elif attr_type == 'Pos':
                    events_attrs[ann['tag_ref']]['pos'] = value

        return events_attrs

    def extract_temporal_entities(self, ann_dict):
        """
        Extract temporal entities (dates, times) from annotations.

        Returns:
            dict: {time_id: {'value': ..., 'function': ...}}
        """
        temporal_entities = {}

        for ann in ann_dict:
            if ann['ann_type'] == 'TextBound' and ann.get('attribute') == 'Time':
                time_id = ann['tag_id']
                temporal_entities[time_id] = {
                    'tag_id': time_id,
                    'value': ann['value'],
                    'function': None
                }

        # Get temporal functions
        for ann in ann_dict:
            if ann['ann_type'] == 'Attribute' and ann['tag_ref'] in temporal_entities:
                if ann['attr_type'] == 'TemporalFunction':
                    temporal_entities[ann['tag_ref']]['function'] = ann['value']

        return temporal_entities

    def get_event_variable(self, event_id, dexpr_list):
        """Get the DRS variable assigned to an event."""
        for item in dexpr_list:
            if item[0].get('event_tag_id') == event_id:
                return item[0].get('event_var')
        return None

    # ========== RULE 1: Tense-Based Ordering ==========
    def rule_tense_ordering(self, events_attrs, dexpr_list, ann_dict):
        """
        Rule: Events with different tenses have temporal ordering.

        - Past before Present before Future
        - Multiple past events: default to narrative order (left-to-right)

        Justification: Reichenbach's temporal framework (E, R, S)
        """
        inferred = []

        event_list = sorted(events_attrs.items(), key=lambda x: x[0])

        for i, (e1_id, e1_attrs) in enumerate(event_list):
            for e2_id, e2_attrs in event_list[i + 1:]:
                t1 = e1_attrs.get('tense')
                t2 = e2_attrs.get('tense')

                if not t1 or not t2:
                    continue

                var1 = self.get_event_variable(e1_id, dexpr_list)
                var2 = self.get_event_variable(e2_id, dexpr_list)

                if not var1 or not var2:
                    continue

                # Past before Present
                if t1 == 'Past' and t2 == 'Present':
                    inferred.append({
                        'rule': 'tense_ordering',
                        'relation': 'occursBefore',
                        'event1': var1,
                        'event2': var2,
                        'confidence': 0.9,
                        'explanation': f"Past tense ({e1_attrs['value']}) before Present ({e2_attrs['value']})"
                    })

                # Past before Future
                elif t1 == 'Past' and t2 == 'Future':
                    inferred.append({
                        'rule': 'tense_ordering',
                        'relation': 'occursBefore',
                        'event1': var1,
                        'event2': var2,
                        'confidence': 0.9,
                        'explanation': f"Past tense ({e1_attrs['value']}) before Future ({e2_attrs['value']})"
                    })

                # Present before Future
                elif t1 == 'Present' and t2 == 'Future':
                    inferred.append({
                        'rule': 'tense_ordering',
                        'relation': 'occursBefore',
                        'event1': var1,
                        'event2': var2,
                        'confidence': 0.8,
                        'explanation': f"Present tense ({e1_attrs['value']}) before Future ({e2_attrs['value']})"
                    })

        return inferred

    # ========== RULE 2: Aspect and Duration ==========
    def rule_aspect_duration(self, events_attrs, dexpr_list, ann_dict):
        """
        Rule: Perfective aspect implies bounded/completed event.
        Progressive aspect implies ongoing, unbounded event.

        - Perfective events are more likely to be sequential
        - Progressive events can overlap with other events

        Justification: Smith's aspectual framework
        """
        inferred = []

        event_list = sorted(events_attrs.items(), key=lambda x: x[0])

        for i, (e1_id, e1_attrs) in enumerate(event_list):
            for e2_id, e2_attrs in event_list[i + 1:]:
                aspect1 = e1_attrs.get('aspect')
                aspect2 = e2_attrs.get('aspect')

                if not aspect1 or not aspect2:
                    continue

                var1 = self.get_event_variable(e1_id, dexpr_list)
                var2 = self.get_event_variable(e2_id, dexpr_list)

                if not var1 or not var2:
                    continue

                # Two perfective events in sequence (narrative progression)
                if aspect1 == 'Perfective' and aspect2 == 'Perfective':
                    inferred.append({
                        'rule': 'aspect_duration',
                        'relation': 'occursBefore',
                        'event1': var1,
                        'event2': var2,
                        'confidence': 0.7,
                        'explanation': f"Sequential perfective events: {e1_attrs['value']} → {e2_attrs['value']}"
                    })

                # Progressive can overlap with perfective
                elif aspect1 == 'Progressive' and aspect2 == 'Perfective':
                    inferred.append({
                        'rule': 'aspect_duration',
                        'relation': 'overlaps',
                        'event1': var1,
                        'event2': var2,
                        'confidence': 0.6,
                        'explanation': f"Progressive ({e1_attrs['value']}) overlaps Perfective ({e2_attrs['value']})"
                    })

        return inferred

    # ========== RULE 3: Event Type Temporal Properties ==========
    def rule_event_type_temporal(self, events_attrs, dexpr_list, ann_dict):
        """
        Rule: Event types have inherent temporal properties.

        - TRANSITION: bounded, implies change of state (BEFORE/AFTER distinction)
        - PROCESS: durative, can overlap with other events
        - STATE: durative, stable, can overlap extensively

        Justification: Vendler's event classification
        """
        inferred = []

        event_list = sorted(events_attrs.items(), key=lambda x: x[0])

        for i, (e1_id, e1_attrs) in enumerate(event_list):
            for e2_id, e2_attrs in event_list[i + 1:]:
                type1 = e1_attrs.get('type')
                type2 = e2_attrs.get('type')

                if not type1 or not type2:
                    continue

                var1 = self.get_event_variable(e1_id, dexpr_list)
                var2 = self.get_event_variable(e2_id, dexpr_list)

                if not var1 or not var2:
                    continue

                # TRANSITION followed by STATE (result state)
                if type1 == 'TRANSITION' and type2 == 'STATE':
                    inferred.append({
                        'rule': 'event_type_temporal',
                        'relation': 'occursBeforeAndLeadsTo',
                        'event1': var1,
                        'event2': var2,
                        'confidence': 0.75,
                        'explanation': f"TRANSITION ({e1_attrs['value']}) leads to STATE ({e2_attrs['value']})"
                    })

                # Two TRANSITIONS in sequence
                elif type1 == 'TRANSITION' and type2 == 'TRANSITION':
                    inferred.append({
                        'rule': 'event_type_temporal',
                        'relation': 'occursBefore',
                        'event1': var1,
                        'event2': var2,
                        'confidence': 0.7,
                        'explanation': f"Sequential TRANSITIONS: {e1_attrs['value']} → {e2_attrs['value']}"
                    })

                # STATE can overlap with PROCESS
                elif type1 == 'STATE' and type2 == 'PROCESS':
                    inferred.append({
                        'rule': 'event_type_temporal',
                        'relation': 'overlaps',
                        'event1': var1,
                        'event2': var2,
                        'confidence': 0.65,
                        'explanation': f"STATE ({e1_attrs['value']}) overlaps PROCESS ({e2_attrs['value']})"
                    })

                # PROCESS can precede TRANSITION
                elif type1 == 'PROCESS' and type2 == 'TRANSITION':
                    inferred.append({
                        'rule': 'event_type_temporal',
                        'relation': 'occursBefore',
                        'event1': var1,
                        'event2': var2,
                        'confidence': 0.65,
                        'explanation': f"PROCESS ({e1_attrs['value']}) before TRANSITION ({e2_attrs['value']})"
                    })

        return inferred

    # ========== RULE 4: Transition Consequence ==========
    def rule_transition_consequence(self, events_attrs, dexpr_list, ann_dict):
        """
        Rule: TRANSITION events have a consequent state.

        A TRANSITION event (achievement/accomplishment) results in a new state.
        If a STATE follows a TRANSITION, they are likely causally/temporally related.

        Justification: Telicity and result states (Moens & Steedman 1988)
        """
        inferred = []

        event_list = sorted(events_attrs.items(), key=lambda x: x[0])

        for i, (e_id, e_attrs) in enumerate(event_list):
            if e_attrs.get('type') != 'TRANSITION':
                continue

            var_trans = self.get_event_variable(e_id, dexpr_list)
            if not var_trans:
                continue

            # Look for subsequent states
            for j in range(i + 1, min(i + 3, len(event_list))):  # Check next 2 events
                next_id, next_attrs = event_list[j]
                if next_attrs.get('type') == 'STATE':
                    var_state = self.get_event_variable(next_id, dexpr_list)
                    if var_state:
                        inferred.append({
                            'rule': 'transition_consequence',
                            'relation': 'resultState',
                            'event1': var_trans,
                            'event2': var_state,
                            'confidence': 0.7,
                            'explanation': f"TRANSITION ({e_attrs['value']}) results in STATE ({next_attrs['value']})"
                        })

        return inferred

    # ========== RULE 5: State Overlap ==========
    def rule_state_overlap(self, events_attrs, dexpr_list, ann_dict):
        """
        Rule: STATE events tend to overlap with other events.

        States are stable conditions that can hold throughout other events.

        Justification: Stative verbs have no natural endpoint
        """
        inferred = []

        event_list = list(events_attrs.items())

        for e1_id, e1_attrs in event_list:
            if e1_attrs.get('type') != 'STATE':
                continue

            var1 = self.get_event_variable(e1_id, dexpr_list)
            if not var1:
                continue

            for e2_id, e2_attrs in event_list:
                if e1_id == e2_id:
                    continue

                e2_type = e2_attrs.get('type')
                if e2_type in ['PROCESS', 'TRANSITION']:
                    var2 = self.get_event_variable(e2_id, dexpr_list)
                    if var2:
                        inferred.append({
                            'rule': 'state_overlap',
                            'relation': 'during',
                            'event1': var2,  # Event happens during state
                            'event2': var1,
                            'confidence': 0.6,
                            'explanation': f"{e2_type} ({e2_attrs['value']}) during STATE ({e1_attrs['value']})"
                        })

        return inferred

    # ========== RULE 6: Process Duration ==========
    def rule_process_duration(self, events_attrs, dexpr_list, ann_dict):
        """
        Rule: PROCESS events have duration and can overlap.

        Processes are activities without inherent endpoint, can co-occur.

        Justification: Atelic events (Vendler 1957)
        """
        inferred = []

        event_list = list(events_attrs.items())
        process_events = [(eid, attrs) for eid, attrs in event_list
                          if attrs.get('type') == 'PROCESS']

        for i, (e1_id, e1_attrs) in enumerate(process_events):
            for e2_id, e2_attrs in process_events[i + 1:]:
                var1 = self.get_event_variable(e1_id, dexpr_list)
                var2 = self.get_event_variable(e2_id, dexpr_list)

                if not var1 or not var2:
                    continue

                inferred.append({
                    'rule': 'process_duration',
                    'relation': 'overlaps',
                    'event1': var1,
                    'event2': var2,
                    'confidence': 0.5,
                    'explanation': f"PROCESS events can overlap: {e1_attrs['value']} ~ {e2_attrs['value']}"
                })

        return inferred

    # ========== RULE 7: Perfective Completion ==========
    def rule_perfective_completion(self, events_attrs, dexpr_list, ann_dict):
        """
        Rule: Perfective aspect + Past tense = completed event before speech time.

        This anchors events to the timeline relative to 'now'.

        Justification: Reichenbach (1947) - E before S
        """
        inferred = []

        for e_id, e_attrs in events_attrs.items():
            if e_attrs.get('aspect') == 'Perfective' and e_attrs.get('tense') == 'Past':
                var = self.get_event_variable(e_id, dexpr_list)
                if var:
                    inferred.append({
                        'rule': 'perfective_completion',
                        'relation': 'completedBeforeNow',
                        'event1': var,
                        'event2': 'NOW',  # Special reference point
                        'confidence': 0.95,
                        'explanation': f"Perfective Past ({e_attrs['value']}) completed before speech time"
                    })

        return inferred

    # ========== RULE 8: Participle Result State ==========
    def rule_participle_result_state(self, events_attrs, dexpr_list, ann_dict):
        """
        Rule: Past participles often denote result states.

        "The book, written in 2020..." - 'written' is a state resulting from writing.

        Justification: Participial constructions (Kratzer 1994)
        """
        inferred = []

        for e_id, e_attrs in events_attrs.items():
            if e_attrs.get('vform') == 'Participle' and e_attrs.get('tense') == 'Past':
                var = self.get_event_variable(e_id, dexpr_list)
                if var:
                    inferred.append({
                        'rule': 'participle_result_state',
                        'relation': 'isResultState',
                        'event1': var,
                        'event2': None,
                        'confidence': 0.75,
                        'explanation': f"Past participle ({e_attrs['value']}) denotes result state"
                    })

        return inferred

    # ========== RULE 9: Temporal Anchor ==========
    def rule_temporal_anchor(self, events_attrs, dexpr_list, ann_dict):
        """
        Rule: Events mentioned with explicit temporal expressions are anchored to that time.

        Uses temporal entity annotations to anchor events.
        """
        inferred = []
        temporal_entities = self.extract_temporal_entities(ann_dict)

        # Find TLINK relations between events and times
        for ann in ann_dict:
            if ann['ann_type'] == 'Relation' and 'TLINK' in ann['rel_type']:
                ref1 = ann['tag_ref1']
                ref2 = ann['tag_ref2']

                # Check if one is a temporal entity
                if ref1 in temporal_entities and ref2 in events_attrs:
                    var_event = self.get_event_variable(ref2, dexpr_list)
                    time_value = temporal_entities[ref1]['value']
                    if var_event:
                        inferred.append({
                            'rule': 'temporal_anchor',
                            'relation': 'at_time',
                            'event1': var_event,
                            'event2': time_value,
                            'confidence': 1.0,
                            'explanation': f"Event ({events_attrs[ref2]['value']}) at time {time_value}"
                        })

                elif ref2 in temporal_entities and ref1 in events_attrs:
                    var_event = self.get_event_variable(ref1, dexpr_list)
                    time_value = temporal_entities[ref2]['value']
                    if var_event:
                        inferred.append({
                            'rule': 'temporal_anchor',
                            'relation': 'at_time',
                            'event1': var_event,
                            'event2': time_value,
                            'confidence': 1.0,
                            'explanation': f"Event ({events_attrs[ref1]['value']}) at time {time_value}"
                        })

        return inferred

    # ========== RULE 10: Narrative Progression ==========
    def rule_narrative_progression(self, events_attrs, dexpr_list, ann_dict):
        """
        Rule: In narrative text, events in the same tense follow discourse order.

        "John entered. He sat down. Mary arrived." → Sequential interpretation

        Justification: Discourse Representation Theory (Kamp & Reyle 1993)
        """
        inferred = []

        # Group events by tense
        events_by_tense = defaultdict(list)
        for e_id, e_attrs in sorted(events_attrs.items(), key=lambda x: x[0]):
            tense = e_attrs.get('tense')
            if tense:
                events_by_tense[tense].append((e_id, e_attrs))

        # Within each tense group, impose narrative order
        for tense, event_list in events_by_tense.items():
            if len(event_list) < 2:
                continue

            for i in range(len(event_list) - 1):
                e1_id, e1_attrs = event_list[i]
                e2_id, e2_attrs = event_list[i + 1]

                # Only for foreground events (TRANSITION/PROCESS)
                if e1_attrs.get('type') in ['TRANSITION', 'PROCESS'] and \
                        e2_attrs.get('type') in ['TRANSITION', 'PROCESS']:

                    var1 = self.get_event_variable(e1_id, dexpr_list)
                    var2 = self.get_event_variable(e2_id, dexpr_list)

                    if var1 and var2:
                        inferred.append({
                            'rule': 'narrative_progression',
                            'relation': 'occursBefore',
                            'event1': var1,
                            'event2': var2,
                            'confidence': 0.65,
                            'explanation': f"Narrative order: {e1_attrs['value']} → {e2_attrs['value']}"
                        })

        return inferred

    # ========== RULE 11: Causal-Temporal Inference ==========
    def rule_causal_temporal(self, events_attrs, dexpr_list, ann_dict):
        """
        Rule: Certain semantic relations imply temporal ordering.

        - Causation implies temporal precedence (cause before effect)
        - Agent-patient relations in TRANSITIONs imply sequencing

        Justification: Causal reasoning (Hobbs 1985)
        """
        inferred = []

        # Look for causal markers in semantic relations
        for ann in ann_dict:
            if ann['ann_type'] == 'Relation' and 'SRLINK' in ann['rel_type']:
                # Extract semantic role type
                rel_parts = ann['rel_type'].split('_')
                if len(rel_parts) > 1:
                    role = rel_parts[1]

                    # Roles that imply temporal precedence
                    if role in ['cause', 'agent', 'instrument']:
                        ref1 = ann['tag_ref1']
                        ref2 = ann['tag_ref2']

                        # Determine which is the event
                        e_ref = ref1 if ref1 in events_attrs else (ref2 if ref2 in events_attrs else None)

                        if e_ref:
                            var = self.get_event_variable(e_ref, dexpr_list)
                            if var:
                                inferred.append({
                                    'rule': 'causal_temporal',
                                    'relation': 'has_role',
                                    'event1': var,
                                    'event2': role,
                                    'confidence': 0.8,
                                    'explanation': f"Event ({events_attrs[e_ref]['value']}) has {role} role"
                                })

        return inferred

    def apply_all_rules(self, ann_dict, dexpr_list):
        """
        Apply all linguistic rules and collect inferred relations.

        Returns:
            list: All inferred temporal relations with metadata
        """
        events_attrs = self.extract_event_attributes(ann_dict)

        all_inferred = []

        for rule_name, rule_func in self.rules.items():
            try:
                inferred = rule_func(events_attrs, dexpr_list, ann_dict)
                all_inferred.extend(inferred)
                print(f"✓ {rule_name}: {len(inferred)} relations inferred")
            except Exception as e:
                print(f"✗ {rule_name}: Error - {e}")

        self.inferred_relations = all_inferred
        return all_inferred

    def filter_by_confidence(self, min_confidence=0.6):
        """Filter inferred relations by confidence threshold."""
        return [rel for rel in self.inferred_relations if rel['confidence'] >= min_confidence]

    def add_to_drs(self, dexpr_list, inferred_relations, min_confidence=0.6):
        """
        Add inferred temporal relations to DRS expressions.

        Args:
            dexpr_list: List of DRS expressions
            inferred_relations: List of inferred relations
            min_confidence: Minimum confidence threshold

        Returns:
            Updated dexpr_list with new relations
        """
        filtered = self.filter_by_confidence(min_confidence)

        for relation in filtered:
            event_var = relation['event1']
            rel_type = relation['relation']

            # Find the event in dexpr_list
            for i, item in enumerate(dexpr_list):
                if item[0].get('event_var') == event_var:
                    n = len(item) - 2

                    # Create appropriate DRS expression based on relation type
                    if rel_type == 'occursBefore':
                        new_expr = dexpr(f'occursBefore({relation["event1"]},{relation["event2"]})')
                    elif rel_type == 'occursAfter':
                        new_expr = dexpr(f'occursAfter({relation["event1"]},{relation["event2"]})')
                    elif rel_type == 'overlaps':
                        new_expr = dexpr(f'overlaps({relation["event1"]},{relation["event2"]})')
                    elif rel_type == 'during':
                        new_expr = dexpr(f'during({relation["event1"]},{relation["event2"]})')
                    elif rel_type in ['resultState', 'occursBeforeAndLeadsTo']:
                        new_expr = dexpr(f'resultState({relation["event1"]},{relation["event2"]})')
                    elif rel_type == 'at_time':
                        new_expr = dexpr(f'at_time({relation["event1"]},"{relation["event2"]}")')
                    elif rel_type == 'completedBeforeNow':
                        new_expr = dexpr(f'completedBefore({relation["event1"]},now)')
                    elif rel_type == 'isResultState':
                        new_expr = dexpr(f'resultStateOf({relation["event1"]})')
                    else:
                        continue  # Skip unknown relation types

                    # Insert the new expression
                    item = item[:n] + (new_expr,) + item[n:]
                    dexpr_list[i] = item
                    break

        print(f"\n✓ Added {len(filtered)} temporal relations to DRS (confidence >= {min_confidence})")
        return dexpr_list

    def generate_report(self, output_file=None):
        """
        Generate a human-readable report of inferred relations.
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("TEMPORAL INFERENCE ENGINE REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"\nTotal inferred relations: {len(self.inferred_relations)}")

        # Group by rule
        by_rule = defaultdict(list)
        for rel in self.inferred_relations:
            by_rule[rel['rule']].append(rel)

        report_lines.append(f"\nRelations by rule:")
        for rule_name, relations in sorted(by_rule.items()):
            report_lines.append(f"  {rule_name}: {len(relations)}")

        # Group by relation type
        by_type = defaultdict(list)
        for rel in self.inferred_relations:
            by_type[rel['relation']].append(rel)

        report_lines.append(f"\nRelations by type:")
        for rel_type, relations in sorted(by_type.items()):
            report_lines.append(f"  {rel_type}: {len(relations)}")

        # Sample relations
        report_lines.append(f"\n{'=' * 80}")
        report_lines.append("SAMPLE INFERRED RELATIONS")
        report_lines.append("=" * 80)

        for i, rel in enumerate(self.inferred_relations[:20]):  # Show first 20
            report_lines.append(f"\n[{i + 1}] {rel['relation']}")
            report_lines.append(f"    Events: {rel['event1']} -> {rel['event2']}")
            report_lines.append(f"    Rule: {rel['rule']}")
            report_lines.append(f"    Confidence: {rel['confidence']:.2f}")
            report_lines.append(f"    Explanation: {rel['explanation']}")

        report = "\n".join(report_lines)

        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"\n✓ Report saved to {output_file}")

        return report


# ========== UTILITY FUNCTIONS ==========

def integrate_inference_engine(ann_dict, dexpr_list, min_confidence=0.6, report_file=None):
    """
    Main integration function to add to brat2drs.py

    Args:
        ann_dict: Parsed BRAT annotations
        dexpr_list: List of DRS expressions
        min_confidence: Minimum confidence for relations (default 0.6)
        report_file: Optional file path to save inference report

    Returns:
        Updated dexpr_list with inferred temporal relations
    """
    engine = TemporalInferenceEngine()

    print("\n" + "=" * 80)
    print("APPLYING TEMPORAL INFERENCE RULES")
    print("=" * 80)

    # Apply all rules
    inferred = engine.apply_all_rules(ann_dict, dexpr_list)

    # Add to DRS
    dexpr_list = engine.add_to_drs(dexpr_list, inferred, min_confidence)

    # Generate report
    if report_file:
        engine.generate_report(report_file)

    return dexpr_list, inferred
