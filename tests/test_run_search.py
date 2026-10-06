from scripts.run_model_improvement import FINAL_JSON, run_final, run_search


def test_execute_search():
    run_search()


def test_execute_final():
    if not FINAL_JSON.exists():
        run_final()
    assert FINAL_JSON.exists()

