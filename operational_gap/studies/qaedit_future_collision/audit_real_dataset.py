"""QAEdit real-source profile (no model execution, no raw question text emitted)."""
import argparse
import collections
import hashlib
import json
import statistics
from pathlib import Path
from prepare_manifest import clean_record, load_rows, allocate


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--report", required=True)
    args=p.parse_args()
    rows, filehash=load_rows(args.input)
    counts=collections.Counter()
    sources=collections.Counter()
    subjects=collections.Counter()
    lengths=collections.Counter()
    invalid_samples=collections.Counter()
    locality_shapes=collections.Counter()
    for r in rows:
        if not isinstance(r,dict):
            counts["not_dictionary"]+=1
            continue
        counts["records"]+=1
        lengths[len(r)]+=1
        locality=r.get("locality")
        locality_shapes[type(locality).__name__+(f":{len(locality)}" if isinstance(locality,list) else "")]+=1
        s=r.get("Source",r.get("source","<missing>"))
        sources[str(s)]+=1
        for k in ("Source","prompt","target","subject","rephrase","locality"):
            if not r.get(k):
                invalid_samples[k]+=1
        c=clean_record(r)
        if c:
            counts["valid_rows"]+=1
            subjects[c["subject_key"]]+=1
        else:
            counts["failed_current_contract"]+=1
    pools, summary=allocate(rows)
    report={
      "data_source":"WanliYoung/Revisit-Editing-Evaluation data/QAEdit.json",
      "source_bytes":Path(args.input).stat().st_size,
      "sha256":filehash,
      "source_schema":sorted({key for r in rows if isinstance(r,dict) for key in r.keys()}),
      "actual_row_count":len(rows),
      "by_source":dict(sources),
      "locality_shapes":dict(locality_shapes),
      "required_field_empty":dict(invalid_samples),
      "duplicate_subject_rows":sum(n for n in subjects.values() if n>1),
      "distinct_subjects":len(subjects),
      "largest_subject_group":max(subjects.values(),default=0),
      "contract_fit":summary,
      "sample_content_in_logs":False
    }
    Path(args.report).write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("QAEdit real source audit:")
    print(json.dumps(report,indent=2,sort_keys=True))
    if summary["valid_unique_rows"]<100:
        raise RuntimeError("Dataset insufficient for precommitted role sizes")
    if summary["cross_role_prompt_or_rephrase_leakage"]!=0:
        raise RuntimeError("Cross-role leakage detected")


if __name__=="__main__":
    main()
