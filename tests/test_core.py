import json

import torch
from PIL import Image, ImageDraw

from glyph_codebook.data import NIST19EpisodeDataset, build_manifest_from_sd19_trees
from glyph_codebook.models import AttentionGlyphGenerator, GlyphCodebook
from glyph_codebook.cli.train_encoder import supervised_contrastive
from glyph_codebook.clustering import spherical_kmeans
from glyph_codebook.metrics import frechet_distance, polynomial_kid


def test_spherical_kmeans_separates_two_directions():
    features = torch.tensor([[1.0, 0.0], [.9, .1], [-1.0, 0.0], [-.9, -.1]])
    centers, labels = spherical_kmeans(features, 2, seed=3)
    assert centers.shape == (2, 2)
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]


def test_supcon_and_distribution_metrics_are_finite():
    features = torch.nn.functional.normalize(torch.randn(8, 16), dim=-1)
    loss = supervised_contrastive(features, torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]), .1)
    assert torch.isfinite(loss)
    assert abs(float(frechet_distance(features, features))) < 1e-4
    assert abs(float(polynomial_kid(features, features, subset_size=8, subsets=1))) < .2


def _model(value_mode):
    keys = torch.randn(3, 4, 16)
    medoids = torch.randn(3, 4, 1, 32, 32).clamp(-1, 1)
    return AttentionGlyphGenerator(GlyphCodebook(keys, medoids, "abc"), embedding_dim=16, hidden_dim=32,
                                   value_mode=value_mode, temperature=.2)


def test_qkv_selector_pixel_and_feature_shapes():
    references = torch.randn(2, 3, 1, 32, 32); characters = torch.tensor([0, 2])
    for mode in ("pixel", "feature"):
        output = _model(mode)(references, characters)
        assert output["image"].shape == (2, 1, 32, 32)
        assert output["attention"].shape == (2, 4)
        assert torch.allclose(output["attention"].sum(-1), torch.ones(2), atol=1e-5)


def test_oracle_attention_is_one_hot():
    output = _model("pixel")(torch.randn(2, 2, 1, 32, 32), torch.tensor([0, 1]), torch.tensor([3, 1]))
    assert torch.equal(output["attention"].argmax(-1), torch.tensor([3, 1]))
    assert torch.equal(output["attention"].sum(-1), torch.ones(2))


def test_episode_references_exclude_target_character(tmp_path):
    records = []
    for char in "abc":
        path = tmp_path / f"w0_{char}.png"; image = Image.new("L", (32, 32), 255); ImageDraw.Draw(image).text((8, 8), char, fill=0); image.save(path)
        records.append({"path": str(path), "character": char, "writer_id": "w0", "sample_id": path.stem, "cluster_id": 0})
    manifest = tmp_path / "data.json"; manifest.write_text(json.dumps(records), encoding="utf-8")
    dataset = NIST19EpisodeDataset(manifest, references=4)
    item = dataset[0]
    assert item["references"].shape == (4, 1, 32, 32)
    assert item["character"] == "a"


def test_official_tree_cross_index_recovers_writer_and_class(tmp_path):
    by_write = tmp_path / "by_write" / "hsf_0" / "f0001_02" / "u0001_02"
    by_class = tmp_path / "by_class" / "41" / "hsf_0"
    by_write.mkdir(parents=True); by_class.mkdir(parents=True)
    image = Image.new("L", (8, 8), 255); ImageDraw.Draw(image).line((1, 7, 4, 1, 7, 7), fill=0, width=1)
    image.save(by_write / "u0001_02_00000.png"); image.save(by_class / "hsf_0_00000.png")
    records, summary = build_manifest_from_sd19_trees(tmp_path / "by_write", tmp_path / "by_class")
    assert summary["indexed"] == 1
    assert records[0]["writer_id"] == "0001"
    assert records[0]["character"] == "A"
