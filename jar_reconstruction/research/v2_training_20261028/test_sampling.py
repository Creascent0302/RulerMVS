"""CPU检查完整覆盖、有限重放、DDP分片与确定性恢复。"""

from collections import Counter, defaultdict
import math
import unittest

from .sampling import RebalancedFullPassSampler


def make_sampler(lengths=(7, 11, 3), multipliers=(1., 1.5, 5.), **options):
    settings = dict(epoch=2, rank=0, world=2, batch_size=2, accumulation=2,
                    seed=20261028, start_update=0)
    settings.update(options)
    return RebalancedFullPassSampler(lengths, domain_multipliers=multipliers, **settings)


class SamplingTest(unittest.TestCase):
    def test_full_coverage_and_fractional_budget(self):
        lengths, multipliers = (7, 11, 3), (1., 1.5, 5.)
        sampler = make_sampler(lengths, multipliers)
        actual = defaultdict(list)
        for domain, entries in sampler.updates:
            actual[domain].extend(entries)
        for domain, (length, multiplier) in enumerate(zip(lengths, multipliers)):
            counts = sampler.domain_counts[domain]
            requested = math.ceil(length * multiplier)
            expected = math.ceil(requested / 8) * 8
            self.assertEqual(set(index for index, _ in actual[domain]), set(range(length)))
            self.assertEqual(counts["requested_exposures"], requested)
            self.assertEqual(counts["total_exposures"], expected)
            self.assertEqual(len(actual[domain]), expected)
            self.assertEqual(sampler.repeats[domain], expected - requested)

    def test_cycles_have_no_replacement_and_different_visits(self):
        # global batch=1，不补齐，明确区分真正重放与padding。
        sampler = make_sampler((5,), (2.4,), world=1, batch_size=1, accumulation=1)
        cycles = defaultdict(list)
        for _, entries in sampler.updates:
            for index, visit in entries:
                cycles[visit].append(index)
        self.assertEqual(sorted(cycles), [6, 7, 8])
        self.assertEqual([len(cycles[k]) for k in sorted(cycles)], [5, 5, 2])
        for entries in cycles.values():
            self.assertEqual(len(entries), len(set(entries)))
        visits_by_index = defaultdict(set)
        for visit, indices in cycles.items():
            for index in indices:
                visits_by_index[index].add(visit)
        self.assertTrue(all(len(visits) >= 2 for visits in visits_by_index.values()))

    def test_eight_rank_accumulation_reconstructs_global_updates(self):
        samplers = [make_sampler((137, 15), (1.5, 4.), world=8, rank=rank,
                                 batch_size=4, accumulation=2) for rank in range(8)]
        streams = [list(sampler) for sampler in samplers]
        for sampler in samplers[1:]:
            self.assertEqual(sampler.updates, samplers[0].updates)
        for update, (domain, entries) in enumerate(samplers[0].updates):
            combined = []
            for micro in range(2):
                for rank in range(8):
                    batch = streams[rank][update * 2 + micro]
                    self.assertEqual(len(batch), 4)
                    self.assertTrue(all(key[0] == domain for key in batch))
                    combined.extend((index, visit) for _, index, visit in batch)
            # 顺序与全局update逐项一致：所有重复均来自显式replay或padding。
            self.assertEqual(combined, entries)
            self.assertEqual(Counter(combined), Counter(entries))

    def test_resume_and_determinism(self):
        full = make_sampler()
        duplicate = make_sampler()
        resumed = make_sampler(start_update=2)
        self.assertEqual(list(full), list(duplicate))
        self.assertEqual(list(resumed), list(full)[2 * full.accumulation:])
        self.assertEqual(len(resumed), len(full) - 2 * full.accumulation)
        finished = make_sampler(start_update=len(full.updates))
        self.assertEqual(list(finished), [])
        self.assertEqual(len(finished), 0)

    def test_next_epoch_changes_visits_without_reusing_cycle_ids(self):
        first = make_sampler((9,), (1.5,), epoch=0, world=1, batch_size=1, accumulation=1)
        second = make_sampler((9,), (1.5,), epoch=1, world=1, batch_size=1, accumulation=1)
        a = {visit for _, entries in first.updates for _, visit in entries}
        b = {visit for _, entries in second.updates for _, visit in entries}
        self.assertTrue(a.isdisjoint(b))
        self.assertNotEqual(first.updates, second.updates)

    def test_dataset_smaller_than_global_batch(self):
        sampler = make_sampler((1,), (1.,), world=8, batch_size=4, accumulation=2)
        self.assertEqual(sampler.domain_counts[0]["total_exposures"], 64)
        self.assertEqual(sampler.repeats[0], 63)
        self.assertEqual(len(list(sampler)), 2)
        self.assertTrue(all(index == 0 for batch in sampler for _, index, _ in batch))

    def test_balanced_order_only_changes_domain_interleaving(self):
        shuffled = make_sampler()
        explicit = make_sampler(domain_schedule="shuffle")
        balanced = make_sampler(domain_schedule="balanced_prefix")
        self.assertEqual(shuffled.updates, explicit.updates)
        self.assertEqual(shuffled.domain_counts, balanced.domain_counts)
        self.assertEqual(shuffled.repeats, balanced.repeats)
        for domain, length in enumerate((7, 11, 3)):
            original = [entries for d, entries in shuffled.updates if d == domain]
            reordered = [entries for d, entries in balanced.updates if d == domain]
            # 域内顺序、每帧visit_epoch和padding逐项不变，因而不存在新增或漏掉的曝光。
            self.assertEqual(original, reordered)
            self.assertEqual({index for entries in reordered for index, _ in entries},
                             set(range(length)))

    def test_balanced_resume_across_ranks_and_accumulation(self):
        for rank in range(8):
            settings = dict(world=8, rank=rank, batch_size=2, accumulation=2,
                            domain_schedule="balanced_prefix")
            full = make_sampler((137, 15, 4), (1.5, 4., 1.), **settings)
            resumed = make_sampler((137, 15, 4), (1.5, 4., 1.), start_update=2, **settings)
            self.assertEqual(list(resumed), list(full)[4:])
            self.assertEqual(len(resumed), len(full) - 4)
            if rank == 0:
                first_rank = full.updates
            else:
                self.assertEqual(full.updates, first_rank)

    def test_balanced_real_training_prefix(self):
        lengths = (3056, 4185, 597, 9476, 9500, 15184, 600)
        multipliers = (1., 1., 5., 1., 1.5, 1., 5.)
        sampler = make_sampler(lengths, multipliers, epoch=0, world=8, batch_size=4,
                               accumulation=1, domain_schedule="balanced_prefix")
        totals = Counter(domain for domain, _ in sampler.updates)
        seen = Counter()
        maximum_error = 0.
        for step, (domain, _) in enumerate(sampler.updates, start=1):
            seen[domain] += 1
            for d in range(len(lengths)):
                expected = step * totals[d] / len(sampler.updates)
                maximum_error = max(maximum_error, abs(seen[d] - expected))
            if step == 112:
                prefix = seen.copy()
        self.assertEqual(len(sampler.updates), 1633)
        self.assertEqual(seen, totals)
        # 这是本项目七域比例的实测界限，不声称所有任意比例都有同一数学上界。
        self.assertLess(maximum_error, 1.)
        for domain in totals:
            expected = 112 * totals[domain] / len(sampler.updates)
            self.assertLess(abs(prefix[domain] - expected), 1.)
        self.assertEqual(sum(prefix.values()), 112)

    def test_balanced_ties_are_seeded_and_small_domains_finish(self):
        settings = dict(epoch=0, world=1, batch_size=1, accumulation=1,
                        domain_schedule="balanced_prefix")
        first = make_sampler((1, 1, 1, 1), (1., 1., 1., 1.), seed=1, **settings)
        duplicate = make_sampler((1, 1, 1, 1), (1., 1., 1., 1.), seed=1, **settings)
        self.assertEqual(first.updates, duplicate.updates)
        orders = set()
        for seed in range(8):
            sampler = make_sampler((1, 1, 1, 1), (1., 1., 1., 1.), seed=seed, **settings)
            order = tuple(domain for domain, _ in sampler.updates)
            orders.add(order)
            self.assertEqual(set(order), {0, 1, 2, 3})
        self.assertGreater(len(orders), 1)
        small = make_sampler((1, 2, 3), (1., 1., 1.), world=8, batch_size=4,
                             accumulation=2, domain_schedule="balanced_prefix")
        self.assertEqual(len(small.updates), 3)
        self.assertEqual(Counter(domain for domain, _ in small.updates), {0: 1, 1: 1, 2: 1})

    def test_invalid_settings(self):
        for lengths, multipliers in [((0,), (1.,)), ((3,), (.9,)), ((3,), (float("nan"),)),
                                     ((3,), (float("inf"),)), ((3,), ())]:
            with self.subTest(lengths=lengths, multipliers=multipliers):
                with self.assertRaises(ValueError):
                    make_sampler(lengths, multipliers)
        with self.assertRaises(ValueError):
            make_sampler(start_update=-1)
        with self.assertRaises(ValueError):
            make_sampler(rank=2, world=2)
        with self.assertRaises(ValueError):
            make_sampler(domain_schedule="unknown")


if __name__ == "__main__":
    unittest.main()
