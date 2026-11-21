"""Audit experiment-live event JSONs; outputs are detection-window coverage, not verified accuracy.
Usage: python experiments/audit_event_logs.py /path/to/data
Only exact-content duplicate JSONs are collapsed; conflicting versions of a session are preserved.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path


def audit(folder):
    seen, records, identities = {}, [], defaultdict(list)
    for p in sorted(Path(folder).glob('*.json')):
        raw = p.read_bytes()
        m = json.loads(raw)
        s = m.get('session')
        if not isinstance(s, dict) or not isinstance(m.get('events'), list):
            continue
        digest = hashlib.sha256(raw).hexdigest()
        if digest in seen:
            seen[digest]['copies'].append(p.name)
            continue
        events = m['events']
        detections = [e for e in events if e.get('type') == 'detected']
        pairs = m.get('marked_pairs', [])
        pair_rows = []
        for pair in pairs:
            detected = [e['kind'] for e in detections if e.get('pair_id') == pair['id']]
            order = any(k == 'push' and 'pull' in detected[i+1:] for i,k in enumerate(detected))
            pair_rows.append({'id':pair['id'], 'kinds':detected, 'push_present':'push' in detected,
                              'pull_present':'pull' in detected, 'ordered_subsequence':order,
                              'exactly_push_pull':detected == ['push','pull']})
        row = {'file':p.name, 'sha256':digest, 'copies':[], 'session':s,
               'detected_counts':dict(Counter(e['kind'] for e in detections)),
               'outside_pair_detections':sum(e.get('pair_id') is None for e in detections),
               'marked_pair_count':len(pairs), 'pairs':pair_rows,
               'timestamps_reversed':sum(b['t_ms'] < a['t_ms'] for a,b in zip(events,events[1:])),
               'ordered_subsequence_windows':sum(x['ordered_subsequence'] for x in pair_rows),
               'exactly_push_pull_windows':sum(x['exactly_push_pull'] for x in pair_rows),
               'follow_pull_sources':dict(Counter(e.get('source','unspecified') for e in detections)),
               'foreign_events':[e for e in events if e.get('type') == 'foreign'],
               'duration_s':None}
        if s.get('started_at') and s.get('finished_at'):
            row['duration_s']=(datetime.fromisoformat(s['finished_at'].replace('Z','+00:00'))-
                               datetime.fromisoformat(s['started_at'].replace('Z','+00:00'))).total_seconds()
        key=(s.get('started_at'),s.get('who'),s.get('mode'))
        identities[key].append(p.name)
        seen[digest]=row
        records.append(row)
    return {'records':records, 'same_identity_conflicts':[v for v in identities.values() if len(v)>1],
            'warning':'Target counts and marked windows do not prove actual attempts, false positives or accuracy. No idle exposure label is available.'}


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder')
    print(json.dumps(audit(p.parse_args().folder),indent=2,ensure_ascii=False))
