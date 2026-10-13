"""Construct the versioned, original routing-workload corpus and grouped folds."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

MATH = [
    "A box contains 4 red, 5 blue and 3 green balls. Three are drawn without replacement. Find the probability that all three have different colors.",
    "Find the least positive integer n such that n leaves remainder 2 modulo 5, remainder 3 modulo 7, and remainder 4 modulo 9.",
    "A right triangle has legs 9 and 12. Find the inradius and explain the geometric relation you use.",
    "Solve x^2-7x+10=0 and explain why these are all the real solutions.",
    "A fair die is rolled until a six occurs. What is the expected number of rolls? Derive the answer.",
    "Find all integer solutions to 3x+5y=17 and express them using one integer parameter.",
    "Prove that among any six integers, two have a difference divisible by five.",
    "How many distinct permutations of the letters in MISSISSIPPI begin with M? Explain the repeated-letter counting.",
    "A rectangle has area 48 and perimeter 28. Determine its side lengths.",
    "Find the coefficient of x^4 in (1+2x)^7 without expanding the whole polynomial.",
    "Evaluate the sum of the first 40 positive odd integers, and prove the pattern you use.",
    "A sequence obeys a_1=2 and a_(n+1)=3a_n+1. Derive a closed form for a_n.",
    "Determine the remainder when 7^123 is divided by 13 using modular arithmetic.",
    "Find the number of nonnegative integer triples (a,b,c) satisfying a+b+c=12 and a>=2.",
    "Two independent fair coins are tossed four times each. Find the probability that their total numbers of heads agree.",
    "Find the maximum of x(10-x) over 0<=x<=10, and justify the maximum without calculus.",
    "A tank fills in 6 hours through one pipe and empties in 9 hours through another. Starting empty, how long does it take to fill with both open?",
    "Determine whether the series sum from n=1 to infinity of 1/(n(n+1)) converges, and find its sum.",
    "Find the derivative of f(x)=x^2 exp(-x), its critical points, and which are local extrema.",
    "Compute the definite integral from 0 to 1 of 3x^2+2x+1 and explain its geometric meaning.",
    "Find the eigenvalues and one eigenvector for each eigenvalue of the matrix [[2,1],[1,2]].",
    "Solve the system 2x+y-z=3, x-2y+z=0, and 3x+y+2z=10.",
    "A graph has five vertices with degrees 4,3,3,2,2. Does a simple graph with these degrees exist? Give an example or a proof.",
    "How many shortest paths go from (0,0) to (5,4) using unit right/up moves and avoiding the point (2,2)?",
    "For a regular hexagon with side length 3, find the area by decomposing it into triangles.",
    "Find all positive divisors of 360 and use prime factorization to count them without listing every divisor.",
    "If a positive real number satisfies x+1/x=5, find x^3+1/x^3 without solving for x explicitly.",
    "A diagnostic has sensitivity 0.9 and specificity 0.95, and the condition prevalence is 0.02. Compute the probability of the condition given a positive test.",
    "Draw two independent uniform real numbers from [0,1]. What is the probability that their sum is greater than 1.2? Use area.",
    "A polynomial p(x) of degree at most two satisfies p(0)=1, p(1)=4, p(2)=9. Find p(5) and justify uniqueness.",
    "In how many ways can eight people sit around a circular table if rotations count as the same seating but reflections do not?",
    "Show that sqrt(3) is irrational using a contradiction based on divisibility.",
    "A population starts at 200 and grows by 5 percent each year. Write its value after 10 years and explain why linear extrapolation is different.",
    "Find the probability of exactly two aces in a uniformly selected five-card hand from a standard 52-card deck.",
    "Find the minimum of x^2+y^2 subject to x+2y=10. Give a proof using an inequality or geometry."
]
CODE = [
    "Implement a Python generator that merges two sorted iterables of integers, preserving duplicates. Explain its memory use and provide an example.",
    "Implement a Python function that checks whether parentheses, square brackets and braces in a string are correctly nested. Ignore other characters.",
    "Implement a Python function returning the longest common prefix of a list of strings. Handle an empty input list.",
    "Implement a Python binary search returning the first position of a target in a sorted list with duplicates, or -1 if absent.",
    "Implement a Python function that reverses the words in a sentence while normalizing whitespace. Include two edge cases.",
    "Design a Python LRU cache with get and put in O(1) time, including replacement of an existing key and capacity zero.",
    "Implement a Python topological sort for a directed graph given as an adjacency dictionary. Detect cycles and include vertices without outgoing edges.",
    "Implement a Python function that groups anagrams without relying on sorting the entire input. Explain the key and its assumptions.",
    "Write a Python iterator that reads newline-delimited JSON from a text stream and yields valid records with their line numbers. Make malformed-record behavior explicit.",
    "Implement breadth-first search to find a shortest path between two nodes in an unweighted graph. Return None when unreachable.",
    "Implement a Python function that coalesces half-open numeric intervals, defining whether adjacent intervals merge. Show an example.",
    "Write a Python fixed-size sliding-window maximum algorithm using a deque, and explain why it is linear time.",
    "Implement a Python function to detect a cycle in a singly linked list without modifying it or storing visited nodes.",
    "Implement a Python trie supporting insert and prefix lookup. Explain the behavior for the empty string.",
    "Write a Python context manager for atomic replacement of a text file using a temporary file in the same directory. Explain cleanup on exceptions.",
    "Implement Dijkstra's shortest-distance algorithm in Python for nonnegative weighted edges. Handle stale heap entries.",
    "Write a Python function that computes Levenshtein edit distance using only O(min(m,n)) auxiliary memory.",
    "Implement an asyncio concurrency limiter that applies an async function to inputs while preserving output order. Explain exception propagation.",
    "Write a Python function finding the kth largest element with a size-k heap. State valid input conditions and duplicate handling.",
    "Implement a Python function that decodes a simple run-length encoding such as [('a',3),('b',2)]. Reject negative counts.",
    "Write a Python function that parses a comma-separated line with double-quoted fields and escaped double quotes. Explain why splitting on commas fails.",
    "Implement a Python disjoint-set data structure with union by size and path compression. Demonstrate connectivity queries.",
    "Design a Python bounded producer-consumer queue example with graceful shutdown. Explain how all worker threads terminate.",
    "Implement a Python interval-overlap query using sorted starts and binary search. Specify the half-open boundary behavior.",
    "Write a Python function computing the median of a stream using two heaps. State the balancing invariant.",
    "Implement a Python reservoir sampler selecting k items from an iterable of unknown length with a supplied random generator.",
    "Write a Python function serializing a tree to nested dictionaries and detecting unexpected cycles. Explain identity versus value comparison.",
    "Implement a Python TTL cache using an injected monotonic clock. Specify whether get refreshes expiration and how expired entries are removed.",
    "Write a Python function computing strongly connected components using Tarjan's algorithm. Explain the lowlink invariant.",
    "Implement a Python function enumerating unique subsets of a list containing duplicate integers without emitting duplicate subsets.",
    "Write a Python stream-processing function that maintains per-key counts for the last 60 seconds using event timestamps. Define out-of-order-event handling.",
    "Design a retry wrapper for an idempotent Python HTTP operation with exponential backoff, a maximum attempt count and an injected sleeper.",
    "Write a Python function validating IPv4 dotted-decimal addresses without accepting missing octets or numbers above 255. Specify leading-zero behavior.",
    "Implement a Python miniature expression evaluator for integers with +, -, multiplication and parentheses. Do not use eval.",
    "Write a Python function that builds a dependency closure from a package-to-dependencies map and reports missing dependencies and cycles."
]
INCIDENT = [
    "An API's timeout rate rose after workers increased from 4 to 16. RSS grew, two restarts helped briefly, and reverting the count restored service. Write a handoff separating observations from hypotheses and list two checks.",
    "A deployment changed a load balancer health path. Healthy backends were removed, requests returned 503, and restoring the old path fixed availability. Write a concise incident analysis and prevention actions.",
    "A database migration held a table lock for 8 minutes; write requests queued while reads continued. Canceling the migration restored writes. Explain immediate mitigation and a safer migration approach.",
    "A consumer fell behind after a malformed message repeatedly failed parsing. Restarting did not help, while quarantining that message restored progress. Explain retry and dead-letter tradeoffs.",
    "An expired TLS certificate caused a scheduled integration to fail overnight. Internal monitoring stayed green because it checked plaintext HTTP. Summarize the detection gap and repairs.",
    "A new cache key omitted tenant ID. Two tenants observed each other's cached summary, while uncached requests were correct. Describe containment, scope investigation and a prevention control.",
    "Disk usage reached 100 percent because log rotation stopped after a permissions change. Writes failed and service resumed after safe log cleanup. Give a handoff with evidence preservation and follow-up checks.",
    "A retry configuration changed from three attempts to unlimited attempts. An upstream slowdown caused a request storm; disabling retries lowered load. Explain the feedback loop and safe retry policy.",
    "A regional DNS record pointed to an old endpoint after a failover. Some clients recovered quickly and others remained broken for an hour. Separate confirmed facts from TTL and resolver hypotheses.",
    "A service began returning stale prices after a cache invalidation subscriber disconnected. Database values stayed correct; reconnecting the subscriber fixed new requests. Suggest resilience and observability improvements.",
    "A database connection pool leaked connections on an exception path. Latency grew until the pool was exhausted; a process restart reset it. Describe how to reproduce and verify the suspected leak.",
    "A batch job used local time to schedule partitions. On a daylight-saving transition it ran one partition twice and skipped another. Propose a correction and an audit of affected output.",
    "A message producer received acknowledgments before its transaction committed. After a crash, acknowledged events were absent. Explain durability semantics and a verification plan.",
    "A feature flag enabled a new serialization format for half the servers. Old clients rejected responses intermittently. Describe rollback and compatibility testing.",
    "A new index increased database write latency while improving one dashboard query. The replication lag rose from seconds to minutes. Give a balanced diagnosis and rollback decision criteria.",
    "CPU remained low while requests stalled on an exhausted thread pool. Tasks in that pool synchronously waited on other tasks in the same pool. Explain the likely deadlock and a design change.",
    "A container was killed for exceeding its memory limit even though host RAM was free. Metrics sampled once a minute missed short spikes. Explain why host free RAM did not prevent the kill.",
    "A storage client changed from bounded to unbounded multipart uploads. Open file descriptors rose until new connections failed. Propose containment and leak versus concurrency checks.",
    "Two scheduled writers concurrently updated the same object using last-write-wins. One update was lost without an error. Describe a concurrency control and a recovery investigation.",
    "A service's p99 latency doubled after its metric labels included request IDs. The metrics backend consumed more memory and CPU. Explain cardinality and propose safer labels.",
    "A partition in a distributed key-value store made writes unavailable for quorum clients, while a read replica still answered older values. Explain the consistency tradeoff without assuming data loss.",
    "A deploy silently changed an environment variable name. The service started with a default timeout of 100 ms and requests failed. Propose startup validation and a regression check.",
    "A data pipeline's output count dropped when a new parser interpreted empty strings as null. Input volume was unchanged. Specify an audit that distinguishes dropped records from transformed values.",
    "A credential rotation updated the application but not a background job. Interactive requests succeeded while the job returned authorization errors. Give containment and inventory actions.",
    "A proxy buffered streaming responses after an upgrade. Total generation time was unchanged but clients saw no early output. Distinguish throughput from first-visible-response latency.",
    "A schema change renamed a field. Producers deployed before consumers, which then rejected messages. Describe a rollout strategy allowing backward compatibility.",
    "A cloud load balancer distributed traffic unevenly because long-lived connections concentrated on two servers. Autoscaling added servers with little traffic. Explain a diagnostic and redistribution options.",
    "An application used wall-clock time for token expiration checks. A clock correction caused premature expiration. Describe the evidence needed and safer elapsed-time measurement.",
    "A spreadsheet import interpreted IDs as floating-point values, corrupting long numeric identifiers. The source CSV still exists. Plan recovery and validation without treating IDs as numbers.",
    "A backup dashboard reported successful uploads, but restore exercises failed because encryption keys were not retained. Distinguish backup completion from recoverability and propose controls.",
    "A new firewall rule allowed the API's outbound traffic but blocked its DNS queries. Connections to cached IPs worked temporarily. Explain the symptom pattern and verification commands conceptually.",
    "A webhook handler processed duplicate deliveries as separate payments. Delivery logs show retries following slow acknowledgments. Describe idempotency design and reconciliation.",
    "A storage bucket lifecycle rule deleted recent data because timestamps used the upload's metadata field instead of object creation time. Give an incident investigation and a safer policy check.",
    "A compaction job saturated disk I/O, increasing database response times while CPU and network utilization stayed modest. Propose attribution measurements and scheduling controls.",
    "A code change swallowed exceptions and returned an empty successful response. Error-rate monitoring improved while user complaints increased. Explain how to detect false success and verify a fix."
]


def build(output: Path):
    rows = []
    for workload, prompts in [("math", MATH), ("code", CODE), ("incident", INCIDENT)]:
        if len(prompts) != 35:
            raise ValueError(f"{workload}: expected 35 distinct tasks")
        for index, prompt in enumerate(prompts):
            request_id = f"{workload}_{index:03d}"
            rows.append({"request_id": request_id, "group_id": request_id, "workload": workload,
                         "source_id": "local-curated-with-assistant", "source_revision": "20261006-v1",
                         "source_item_id": request_id, "license": "repository MIT",
                         "prompt_text": prompt, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                         "reference_answer_or_rubric": None, "pilot_flag": index < 5})
    if len({row["prompt_sha256"] for row in rows}) != len(rows):
        raise ValueError("duplicate prompts")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")


def split(rows, seed=20261006):
    groups = {}
    ids = set()
    for row in rows:
        if row["request_id"] in ids:
            raise ValueError("duplicate request ID")
        ids.add(row["request_id"])
        groups.setdefault(row["group_id"], []).append(row)
    for members in groups.values():
        if len({r["pilot_flag"] for r in members}) != 1:
            raise ValueError("pilot and primary share a group")
    rng = random.Random(seed)
    assignment = {}
    balance = {w: [0] * 5 for w in ("math", "code", "incident")}
    primary = [g for g, members in groups.items() if not members[0]["pilot_flag"]]
    rng.shuffle(primary)
    for group in primary:
        weights = {w: sum(r["workload"] == w for r in groups[group]) for w in balance}
        fold = min(range(5), key=lambda f: (sum(balance[w][f] * n for w, n in weights.items()),
                                           sum(balance[w][f] for w in balance), f))
        assignment[group] = fold
        for w, n in weights.items():
            balance[w][fold] += n
    repeat = []
    for workload in balance:
        candidates = sorted(r["request_id"] for r in rows if r["workload"] == workload and not r["pilot_flag"])
        rng.shuffle(candidates)
        repeat.extend(candidates[:3])
    return {"seed": seed, "fold_count": 5, "group_to_fold": assignment,
            "workload_fold_counts": balance, "repeat_request_ids": sorted(repeat)}


if __name__ == "__main__":
    target = Path("experiments/prompts/pruning_requests.jsonl")
    build(target)
    rows = [json.loads(line) for line in target.read_text().splitlines()]
    Path("experiments/prompts/pruning_splits.json").write_text(json.dumps(split(rows), indent=2) + "\n")
