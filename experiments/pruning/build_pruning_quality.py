"""Freeze new synthetic task references and text sequences for actual M75 inference."""
from __future__ import annotations

import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path

from experiments.pruning.moe_pruning import atomic_json, sha256


def tasks():
    rows = []
    for family in range(10):
        for variant in range(10):
            a, b, c = 17 + variant * 3, 11 + family * 2, 5 + variant
            questions = [
                (f"Compute {a} * {b} + {c}.", a*b+c),
                (f"Solve {b}*x + {c} = {a*b+c} for x.", a),
                (f"A rectangle has sides {a} and {b}. What is its area?", a*b),
                (f"A progression starts at {a} and increases by {c} each term. Find its {b}th term.", a+(b-1)*c),
                (f"Find the sum of the first {b} terms of an arithmetic progression starting at {a} with difference {c}.", Fraction(b*(2*a+(b-1)*c), 2)),
                (f"A bag has {a} red balls and {b} blue balls. Draw one uniformly. What is the probability it is red?", Fraction(a, a+b)),
                (f"For a right triangle with legs {3*c} and {4*c}, find the hypotenuse.", 5*c),
                (f"How many unordered pairs can be selected from {a} distinct objects?", a*(a-1)//2),
                (f"A price of {100*a} is reduced by {b} percent. Find the reduced price.", a*(100-b)),
                (f"Find the remainder when {a}**2 + {c} is divided by {b}.", (a*a+c)%b),
            ]
            question, answer = questions[family]
            rows.append(dict(request_id=f"quality_math_{family:02d}_{variant:02d}", group_id=f"math_family_{family}",
                             workload="math", prompt=question+' Return only JSON {"answer": "value"}; use a fraction p/q for noninteger exact answers.',
                             reference={"answer": str(answer)}, scorer="exact_rational_answer"))
    for family in range(10):
        for variant in range(10):
            cases = []
            modulus, shift = variant + 3, variant + 2
            specifications = [
                ("values", f"return the sum of all values divisible by {modulus}", lambda x: sum(v for v in x if v % modulus == 0)),
                ("values", f"return a new list with {shift} added to every value", lambda x: [v+shift for v in x]),
                ("values", "return the distinct values in order of first appearance", lambda x: list(dict.fromkeys(x))),
                ("values", f"return the number of values strictly greater than {shift}", lambda x: sum(v > shift for v in x)),
                ("text", "return True exactly when text equals its reverse; comparisons are case sensitive", lambda x: x == x[::-1]),
                ("values", "return a list of prefix sums of the input values", lambda x: [sum(x[:i+1]) for i in range(len(x))]),
                ("values", "return the second largest distinct value, or None when there are fewer than two distinct values", lambda x: sorted(set(x))[-2] if len(set(x)) > 1 else None),
                ("text", f"rotate the string left by {shift} characters modulo its length; return empty for empty input", lambda x: x[shift%len(x):]+x[:shift%len(x)] if x else ""),
                ("values", f"return True if any two different indices sum to {modulus}", lambda x: any(x[i]+x[j] == modulus for i in range(len(x)) for j in range(i+1,len(x)))),
                ("values", "return the length of the longest contiguous run of equal values, or zero for an empty input", lambda x: max((sum(1 for _ in g) for _,g in __import__('itertools').groupby(x)), default=0)),
            ]
            parameter, specification, reference = specifications[family]
            inputs = (["", "a", "aba", "abba", "abc", "Aa", "xyzxyz", "mississippi"] if parameter == "text" else
                      [[], [0], [1,1], [-3,0,3], [5,2,5,2,9], list(range(-5,6)), [modulus,0,modulus], [shift,shift+1,shift-1], [7]*5+[8]*3])
            for value in inputs:
                cases.append({"input": value, "expected": reference(value)})
            name = f"solve_{family}_{variant}"
            prompt = f"Write Python 3 function {name}({parameter}) to {specification}. Do not mutate the input. Return only the function code, without explanations."
            rows.append(dict(request_id=f"quality_code_{family:02d}_{variant:02d}", group_id=f"code_family_{family}",
                             workload="code", prompt=prompt, reference={"function": name, "cases": cases}, scorer="restricted_functional_pass1"))
    scenarios = [
        ("A deployment is followed by an immediate rise in HTTP 500s; the previous version was healthy.", "rollback_deployment", ["500", "deployment"]),
        ("TLS handshakes fail and the certificate expiration timestamp is in the past.", "renew_certificate", ["certificate", "expir"]),
        ("The database refuses writes with 'No space left on device'; the volume is 100% full.", "restore_disk_headroom", ["space", "full"]),
        ("Connection pool requests time out; all pool slots are occupied by slow queries.", "inspect_slow_queries", ["pool", "queries"]),
        ("The resolver returns NXDOMAIN for the service name; the service is healthy when addressed by IP.", "repair_dns_record", ["nxdomain", "dns"]),
        ("Repeated requests return HTTP 429 and Retry-After: 30; the service enforces a quota.", "backoff_requests", ["429", "quota"]),
        ("A container repeatedly exits with OOMKilled; its RSS reaches its configured memory limit.", "inspect_memory_usage", ["oom", "memory"]),
        ("Authentication returns 'token expired'; the token expiry precedes the server's current time.", "refresh_token", ["token", "expir"]),
        ("A replica serves stale data; replication lag is growing while the primary is healthy.", "route_reads_to_primary", ["replica", "lag"]),
        ("A queue accumulates jobs; consumers have stopped and logs show an invalid credential.", "restore_consumer_credentials", ["consumer", "credential"]),
    ]
    actions = [scenario[1] for scenario in scenarios] + ["delete_production_data", "disable_authentication", "ignore_alert"]
    for family, (scenario, action, terms) in enumerate(scenarios):
        for variant in range(10):
            prompt = (f"Incident INC-{7000+family*100+variant} at service svc-{family}-{variant}: {scenario} "
                      f"Choose the immediate response from {json.dumps(actions)}. "
                      'Return only JSON {"action": "one listed action", "reason": "cite two relevant observations"}.')
            rows.append(dict(request_id=f"quality_incident_{family:02d}_{variant:02d}", group_id=f"incident_family_{family}",
                             workload="incident", prompt=prompt, reference={"action": action, "evidence_terms": terms,
                             "unsafe_actions": ["delete_production_data", "disable_authentication", "ignore_alert"]}, scorer="frozen_structured_incident_rubric"))
    for row in rows:
        row["prompt_sha256"] = hashlib.sha256(row["prompt"].encode()).hexdigest()
    return rows


def sequences():
    rows = []
    templates = [
        "A laboratory measured {a} units of flow at station {s}. Observer {t} recorded the timestamp before adjusting the valve. A second measurement is required to distinguish a persistent change from noise. The control report retains every observation and states which instruments were calibrated.",
        "A shop received {a} cartons for order {s}. Clerk {t} counted the contents before assigning shelf space. Damaged packaging was photographed and the inventory record was reconciled with the delivery note. The manager separated the physical count from the amount promised by the supplier.",
        "A queue processed {a} jobs during interval {t} for service {s}. The dashboard reports arrivals and completed jobs separately. An increasing backlog can reflect a slower consumer or more incoming work. The operator checked both rates before changing concurrency or retry policy.",
        "A database audit recorded transaction {s} with {a} affected rows. Reviewer {t} verified the query and checked whether the transaction committed. Replica lag and primary load are distinct measurements. A repair requires a backup and a documented rollback rather than an unreviewed destructive command.",
        "In exercise {s}, a student arranged {a} counters into equal groups. Step {t} states the operation and its units explicitly. A numerical answer should be checked by substituting it into the original equation. The written solution distinguishes an exact fraction from a rounded decimal approximation.",
        "Function example {s} accepts a list containing {a} integers. Case {t} checks the returned value and whether the original list changed. An empty input and repeated values are part of the specification. The implementation records its time complexity and avoids relying on accidental ordering of a set.",
        "A transport planner assigned {a} packages to route {s}. At checkpoint {t}, the dispatcher verified the arrival window and remaining capacity. A later arrival does not prove that a driver chose an inefficient route. Weather, loading delays, and traffic observations are retained as separate evidence.",
        "A library accession batch {s} contained {a} volumes. Cataloger {t} checked edition identifiers before merging duplicate records. Similar titles can describe different editions. The catalog preserves the original identifiers and maintains a reversible record of every correction.",
        "A network probe sent {a} requests to endpoint {s}. During sample {t}, the monitor recorded response code, latency, and resolver status. A DNS lookup failure differs from an HTTP server error. The incident note identifies the failing layer before proposing a retry or configuration change.",
        "A scheduler assigned {a} minutes to project {s}. Review {t} compared planned milestones with completed work. An estimate is a prediction rather than a guarantee. The team preserves dependencies and records uncertainty before allocating additional capacity to the next task.",
    ]
    for index in range(100):
        paragraphs = []
        for turn in range(16):
            paragraphs.append(templates[index%10].format(a=17+index+turn, s=index+101, t=turn+1))
        rows.append(dict(sequence_id=f"quality_text_{index:03d}", group_id=f"text_template_{index%10}",
                         text="\n\n".join(paragraphs), source="new authored synthetic operational narrative", revision="20261006-m75-v1"))
    return rows


def build(args):
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    rows = tasks()
    if len(rows) != 300 or len({row["prompt_sha256"] for row in rows}) != 300:
        raise ValueError("quality tasks must have 300 unique prompts")
    for name, values in (("tasks.jsonl", rows), ("sequences.jsonl", sequences())):
        (root / name).write_text("".join(json.dumps(row, allow_nan=False)+"\n" for row in values))
    atomic_json(root / "manifest.json", dict(tasks=300, tasks_per_workload=100, sequences=100,
                scored_continuation_tokens_per_sequence=256, minimum_scored_tokens_per_model=25600,
                confidence_interval_unit="template families for tasks; text-template clusters for NLL",
                representativeness="synthetic elementary tasks; not a comprehensive real-world benchmark",
                scorers={"math": "exact rational final answer", "code": "all frozen cases pass in restricted subprocess",
                         "incident": "correct action 0.6; each cited observation 0.2; unsafe choice recorded"},
                hashes={name: sha256(root / name) for name in ("tasks.jsonl", "sequences.jsonl")},
                source_sha256=sha256(Path(__file__))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    build(parser.parse_args())
