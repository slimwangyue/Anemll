#!/bin/bash
# Multi-round conversation validation for Qwen3.5-4B stable LUT6 models
# Tests the model via Swift CLI on ANE to validate before iOS deployment
#
# Usage: ./tests/validate_multiround.sh [--verbose]

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CLI="$REPO_DIR/anemll-swift-cli/.build/release/anemllcli"
META="$REPO_DIR/qwen3_5_stable_models/meta.yaml"

VERBOSE=""
[[ "$1" == "--verbose" ]] && VERBOSE="--debug-level 1"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# Check prerequisites
if [[ ! -f "$CLI" ]]; then
    echo -e "${YELLOW}Building Swift CLI...${NC}"
    cd "$REPO_DIR/anemll-swift-cli" && swift build -c release 2>&1 | tail -3
fi
if [[ ! -f "$META" ]]; then
    echo -e "${RED}ERROR: meta.yaml not found at $META${NC}" && exit 1
fi

PASS=0
FAIL=0
TOTAL=0

run_test() {
    local name="$1"
    local prompt="$2"
    local max_tokens="${3:-100}"
    local expect_pattern="$4"
    TOTAL=$((TOTAL + 1))

    echo -e "\n${CYAN}[$TOTAL] $name${NC}"
    echo "  Prompt: $prompt"

    OUTPUT=$("$CLI" --meta "$META" --prompt "$prompt" --max-tokens "$max_tokens" --template qwen $VERBOSE 2>&1)

    # Extract assistant response
    RESPONSE=$(echo "$OUTPUT" | sed -n '/^Assistant:/,/^$/p' | head -10)
    STATS=$(echo "$OUTPUT" | grep -E "t/s.*TTFT" | tail -1)

    echo "  Response: $RESPONSE"
    echo "  $STATS"

    # Check for errors
    if echo "$OUTPUT" | grep -q "Error during generation"; then
        echo -e "  ${RED}FAIL: Generation error${NC}"
        FAIL=$((FAIL + 1))
        return
    fi

    # Check for EOS (natural stop)
    if echo "$STATS" | grep -q "Stop: eos_token"; then
        echo -e "  ${GREEN}✓ Stopped at EOS${NC}"
    elif echo "$STATS" | grep -q "Stop: max_tokens"; then
        echo -e "  ${YELLOW}⚠ Hit max_tokens (may be OK for long responses)${NC}"
    fi

    # Check expected pattern if provided
    if [[ -n "$expect_pattern" ]]; then
        if echo "$RESPONSE" | grep -iq "$expect_pattern"; then
            echo -e "  ${GREEN}✓ Contains expected: '$expect_pattern'${NC}"
            PASS=$((PASS + 1))
        else
            echo -e "  ${RED}✗ Missing expected: '$expect_pattern'${NC}"
            FAIL=$((FAIL + 1))
        fi
    else
        # No pattern check — just verify it produced output
        if [[ -n "$RESPONSE" ]] && ! echo "$RESPONSE" | grep -q "^Assistant: *$"; then
            echo -e "  ${GREEN}✓ Produced output${NC}"
            PASS=$((PASS + 1))
        else
            echo -e "  ${RED}✗ Empty response${NC}"
            FAIL=$((FAIL + 1))
        fi
    fi
}

echo "============================================="
echo " Qwen3.5-4B LUT6 Multi-Round Validation"
echo " Models: $META"
echo "============================================="

# --- Basic knowledge ---
run_test "Basic math" "What is 2+2?" 50 ""
run_test "Capital city" "What is the capital of France?" 80 "Paris"
run_test "Simple fact" "How many days are in a week?" 50 "7\|seven"

# --- Reasoning ---
run_test "Simple reasoning" "If I have 5 apples and give away 2, how many do I have left?" 80 "3\|three"
run_test "Comparison" "Which is larger, the Sun or the Moon?" 100 "Sun\|sun"

# --- Language tasks ---
run_test "Translation awareness" "Say hello in Spanish" 50 "hola\|Hola"
run_test "List generation" "Name 3 colors" 80 ""
run_test "Definition" "What is photosynthesis?" 120 ""

# --- Longer generation ---
run_test "Story start" "Write a very short story about a cat in 2 sentences." 150 ""
run_test "Explanation" "Explain gravity in simple terms." 150 ""

# --- Edge cases ---
run_test "Single word answer" "Yes or no: Is water wet?" 30 ""
run_test "Number recall" "What year did World War 2 end?" 50 "1945"

echo ""
echo "============================================="
echo -e " Results: ${GREEN}$PASS passed${NC}, ${RED}$FAIL failed${NC}, $TOTAL total"
echo "============================================="

if [[ $FAIL -gt 0 ]]; then
    echo -e "${YELLOW}Some tests failed — review responses above.${NC}"
    exit 1
else
    echo -e "${GREEN}All tests passed!${NC}"
fi
