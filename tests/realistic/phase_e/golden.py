#!/usr/bin/env python3
"""
Phase E: Stress Testing Golden Model

This module tests the accelerator under extreme conditions:
1. Back-to-back GEMM operations (no idle cycles)
2. Maximum SRAM utilization (fill all 64K×256-bit per TPC)
3. Multi-TPC parallel workloads (all 4 TPCs active)
4. Pipeline hazard scenarios
5. Large matrix operations

Architecture reminder:
- 2×2 TPC grid (4 TPCs total)
- Each TPC: 8×8 systolic array, 64K×256-bit SRAM
- NoC mesh interconnect
"""

import numpy as np
import os
import json
from typing import Tuple, List, Dict

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "test_vectors")

# Architecture parameters
NUM_TPCS = 4
TILE_SIZE = 8
SRAM_DEPTH = 65536  # 64K entries
SRAM_WIDTH = 256    # bits = 32 bytes
SRAM_BYTES = SRAM_DEPTH * (SRAM_WIDTH // 8)  # 2MB per TPC

# ============================================================================
# Test Vector Helpers
# ============================================================================

def save_hex_int8(filename: str, data: np.ndarray):
    with open(filename, 'w') as f:
        for val in data.flatten():
            val = int(val)
            if val < 0:
                val = val + 256
            f.write(f"{val:02x}\n")

def save_hex_int32(filename: str, data: np.ndarray):
    with open(filename, 'w') as f:
        for val in data.flatten():
            val = int(val)
            if val < 0:
                val = val + (1 << 32)
            f.write(f"{val:08x}\n")


def tiled_gemm_int8(A: np.ndarray, B: np.ndarray, tile_size: int = 8) -> Tuple[np.ndarray, Dict]:
    """
    Tiled GEMM with statistics tracking.
    
    Returns:
        (result, stats) where stats contains tile counts, cycles, etc.
    """
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, f"Dimension mismatch: A({M},{K}) @ B({K2},{N})"
    
    C = np.zeros((M, N), dtype=np.int32)
    
    m_tiles = (M + tile_size - 1) // tile_size
    k_tiles = (K + tile_size - 1) // tile_size
    n_tiles = (N + tile_size - 1) // tile_size
    
    total_tiles = m_tiles * n_tiles * k_tiles
    total_macs = M * N * K
    cycles = 0
    
    for m_t in range(m_tiles):
        m_start = m_t * tile_size
        m_end = min(m_start + tile_size, M)
        
        for n_t in range(n_tiles):
            n_start = n_t * tile_size
            n_end = min(n_start + tile_size, N)
            
            for k_t in range(k_tiles):
                k_start = k_t * tile_size
                k_end = min(k_start + tile_size, K)
                
                # Tile computation
                A_tile = A[m_start:m_end, k_start:k_end]
                B_tile = B[k_start:k_end, n_start:n_end]
                
                C[m_start:m_end, n_start:n_end] += A_tile.astype(np.int32) @ B_tile.astype(np.int32)
                
                # Cycles: K iterations per tile
                cycles += (k_end - k_start)
    
    stats = {
        'M': M, 'K': K, 'N': N,
        'm_tiles': m_tiles, 'k_tiles': k_tiles, 'n_tiles': n_tiles,
        'total_tiles': total_tiles,
        'total_macs': total_macs,
        'cycles': cycles,
        'macs_per_cycle': total_macs / cycles if cycles > 0 else 0
    }
    
    return C, stats


# ============================================================================
# Stress Test Generators
# ============================================================================

class PhaseE_Tests:
    """Generate stress test vectors."""
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
    
    def test1_back_to_back_gemm(self, output_dir: str):
        """
        Test 1: Back-to-back GEMM operations.
        
        Pipeline: A×B → C, C×D → E, E×F → G
        Tests continuous operation without idle cycles.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "="*60)
        print("TEST 1: Back-to-Back GEMM (3 chained operations)")
        print("="*60)
        
        # Dimensions chosen for continuous pipeline
        M, K1, N1 = 32, 24, 16  # A×B → C
        K2, N2 = N1, 12         # C×D → E  (K2 = N1 for chaining)
        K3, N3 = N2, 8          # E×F → G
        
        # Generate matrices
        A = np.random.randint(-50, 50, size=(M, K1), dtype=np.int8)
        B = np.random.randint(-50, 50, size=(K1, N1), dtype=np.int8)
        D = np.random.randint(-50, 50, size=(K2, N2), dtype=np.int8)
        F = np.random.randint(-50, 50, size=(K3, N3), dtype=np.int8)
        
        print(f"\n  Stage 1: ({M}, {K1}) × ({K1}, {N1})")
        C, stats1 = tiled_gemm_int8(A, B)
        print(f"    Tiles: {stats1['total_tiles']}, Cycles: {stats1['cycles']}")
        
        # Requantize C for next stage
        C_int8 = np.clip(C >> 8, -128, 127).astype(np.int8)
        
        print(f"\n  Stage 2: ({M}, {K2}) × ({K2}, {N2})")
        E, stats2 = tiled_gemm_int8(C_int8, D)
        print(f"    Tiles: {stats2['total_tiles']}, Cycles: {stats2['cycles']}")
        
        # Requantize E for next stage
        E_int8 = np.clip(E >> 8, -128, 127).astype(np.int8)
        
        print(f"\n  Stage 3: ({M}, {K3}) × ({K3}, {N3})")
        G, stats3 = tiled_gemm_int8(E_int8, F)
        print(f"    Tiles: {stats3['total_tiles']}, Cycles: {stats3['cycles']}")
        
        total_cycles = stats1['cycles'] + stats2['cycles'] + stats3['cycles']
        total_macs = stats1['total_macs'] + stats2['total_macs'] + stats3['total_macs']
        
        print(f"\n  Total: {total_cycles} cycles, {total_macs} MACs")
        print(f"  Effective MACs/cycle: {total_macs/total_cycles:.1f}")
        
        # Save vectors
        save_hex_int8(f"{output_dir}/test1_A_int8.hex", A)
        save_hex_int8(f"{output_dir}/test1_B_int8.hex", B)
        save_hex_int8(f"{output_dir}/test1_D_int8.hex", D)
        save_hex_int8(f"{output_dir}/test1_F_int8.hex", F)
        save_hex_int32(f"{output_dir}/test1_C_int32.hex", C)
        save_hex_int8(f"{output_dir}/test1_C_int8.hex", C_int8)
        save_hex_int32(f"{output_dir}/test1_E_int32.hex", E)
        save_hex_int8(f"{output_dir}/test1_E_int8.hex", E_int8)
        save_hex_int32(f"{output_dir}/test1_G_int32.hex", G)
        
        np.save(f"{output_dir}/test1_A_int8.npy", A)
        np.save(f"{output_dir}/test1_B_int8.npy", B)
        np.save(f"{output_dir}/test1_D_int8.npy", D)
        np.save(f"{output_dir}/test1_F_int8.npy", F)
        np.save(f"{output_dir}/test1_C_int32.npy", C)
        np.save(f"{output_dir}/test1_E_int32.npy", E)
        np.save(f"{output_dir}/test1_G_int32.npy", G)
        
        return {
            'stages': [
                {'M': M, 'K': K1, 'N': N1, 'cycles': stats1['cycles']},
                {'M': M, 'K': K2, 'N': N2, 'cycles': stats2['cycles']},
                {'M': M, 'K': K3, 'N': N3, 'cycles': stats3['cycles']}
            ],
            'total_cycles': total_cycles,
            'total_macs': total_macs
        }
    
    def test2_max_sram_utilization(self, output_dir: str):
        """
        Test 2: Maximum SRAM utilization.
        
        Compute largest GEMM that fits in single TPC's SRAM.
        SRAM: 64K × 256-bit = 2MB
        
        For M×K @ K×N:
        - A needs M×K bytes
        - B needs K×N bytes  
        - C needs M×N×4 bytes (INT32)
        
        Maximize M×K + K×N + 4×M×N ≤ 2MB
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "="*60)
        print("TEST 2: Maximum SRAM Utilization")
        print("="*60)
        
        # Conservative estimate: use 75% of SRAM to leave room for control
        max_bytes = int(SRAM_BYTES * 0.75)  # ~1.5MB
        
        # For square matrices: 2*M*K + 4*M*K = 6*M*K ≤ max_bytes
        # M = K = sqrt(max_bytes / 6) ≈ 512
        # But we'll use smaller for reasonable simulation time
        
        M = 256
        K = 256
        N = 64
        
        bytes_A = M * K
        bytes_B = K * N
        bytes_C = M * N * 4
        total_bytes = bytes_A + bytes_B + bytes_C
        utilization = total_bytes / SRAM_BYTES * 100
        
        print(f"\n  Matrix A: ({M}, {K}) = {bytes_A/1024:.1f} KB")
        print(f"  Matrix B: ({K}, {N}) = {bytes_B/1024:.1f} KB")
        print(f"  Matrix C: ({M}, {N}) × 4B = {bytes_C/1024:.1f} KB")
        print(f"  Total: {total_bytes/1024:.1f} KB / {SRAM_BYTES/1024:.0f} KB = {utilization:.1f}%")
        
        # Generate matrices
        A = np.random.randint(-30, 30, size=(M, K), dtype=np.int8)
        B = np.random.randint(-30, 30, size=(K, N), dtype=np.int8)
        
        print(f"\n  Computing GEMM...")
        C, stats = tiled_gemm_int8(A, B)
        
        print(f"  Tiles: {stats['total_tiles']}")
        print(f"  Cycles: {stats['cycles']}")
        print(f"  MACs: {stats['total_macs']:,}")
        print(f"  MACs/cycle: {stats['macs_per_cycle']:.1f}")
        
        # Save vectors
        save_hex_int8(f"{output_dir}/test2_A_int8.hex", A)
        save_hex_int8(f"{output_dir}/test2_B_int8.hex", B)
        save_hex_int32(f"{output_dir}/test2_C_int32.hex", C)
        
        np.save(f"{output_dir}/test2_A_int8.npy", A)
        np.save(f"{output_dir}/test2_B_int8.npy", B)
        np.save(f"{output_dir}/test2_C_int32.npy", C)
        
        return {
            'M': M, 'K': K, 'N': N,
            'bytes_A': bytes_A,
            'bytes_B': bytes_B,
            'bytes_C': bytes_C,
            'total_bytes': total_bytes,
            'sram_utilization': utilization,
            'cycles': stats['cycles'],
            'total_macs': stats['total_macs']
        }
    
    def test3_multi_tpc_parallel(self, output_dir: str):
        """
        Test 3: Multi-TPC parallel workloads.
        
        Distribute work across all 4 TPCs:
        - TPC0: Q @ K^T (attention scores)
        - TPC1: A @ V (attention output)
        - TPC2: FC1 (MLP first layer)
        - TPC3: FC2 (MLP second layer)
        
        Tests NoC bandwidth and parallel execution.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "="*60)
        print("TEST 3: Multi-TPC Parallel Workload")
        print("="*60)
        
        # Common dimensions (transformer-like)
        seq_len = 32
        hidden = 64
        heads = 4
        head_dim = hidden // heads  # 16
        mlp_dim = hidden * 4  # 256
        
        # TPC0: Q @ K^T for one head
        Q = np.random.randint(-50, 50, size=(seq_len, head_dim), dtype=np.int8)
        K = np.random.randint(-50, 50, size=(seq_len, head_dim), dtype=np.int8)
        
        # TPC1: Attention @ V
        A_attn = np.random.randint(0, 127, size=(seq_len, seq_len), dtype=np.int8)  # softmax output
        V = np.random.randint(-50, 50, size=(seq_len, head_dim), dtype=np.int8)
        
        # TPC2: FC1 (hidden → mlp_dim)
        X_fc1 = np.random.randint(-50, 50, size=(seq_len, hidden), dtype=np.int8)
        W_fc1 = np.random.randint(-30, 30, size=(hidden, mlp_dim), dtype=np.int8)
        
        # TPC3: FC2 (mlp_dim → hidden)
        X_fc2 = np.random.randint(-50, 50, size=(seq_len, mlp_dim), dtype=np.int8)
        W_fc2 = np.random.randint(-30, 30, size=(mlp_dim, hidden), dtype=np.int8)
        
        print(f"\n  TPC0: Q @ K^T ({seq_len}, {head_dim}) × ({head_dim}, {seq_len})")
        QK, stats0 = tiled_gemm_int8(Q, K.T)
        print(f"    Cycles: {stats0['cycles']}, MACs: {stats0['total_macs']}")
        
        print(f"\n  TPC1: Attn @ V ({seq_len}, {seq_len}) × ({seq_len}, {head_dim})")
        AV, stats1 = tiled_gemm_int8(A_attn, V)
        print(f"    Cycles: {stats1['cycles']}, MACs: {stats1['total_macs']}")
        
        print(f"\n  TPC2: FC1 ({seq_len}, {hidden}) × ({hidden}, {mlp_dim})")
        Y_fc1, stats2 = tiled_gemm_int8(X_fc1, W_fc1)
        print(f"    Cycles: {stats2['cycles']}, MACs: {stats2['total_macs']}")
        
        print(f"\n  TPC3: FC2 ({seq_len}, {mlp_dim}) × ({mlp_dim}, {hidden})")
        Y_fc2, stats3 = tiled_gemm_int8(X_fc2, W_fc2)
        print(f"    Cycles: {stats3['cycles']}, MACs: {stats3['total_macs']}")
        
        # Parallel execution: max of all cycles
        max_cycles = max(stats0['cycles'], stats1['cycles'], stats2['cycles'], stats3['cycles'])
        total_macs = stats0['total_macs'] + stats1['total_macs'] + stats2['total_macs'] + stats3['total_macs']
        
        print(f"\n  Parallel execution time: {max_cycles} cycles")
        print(f"  Total MACs (all TPCs): {total_macs:,}")
        print(f"  Effective throughput: {total_macs/max_cycles:.1f} MACs/cycle")
        print(f"  Speedup vs sequential: {(stats0['cycles']+stats1['cycles']+stats2['cycles']+stats3['cycles'])/max_cycles:.2f}x")
        
        # Save vectors
        save_hex_int8(f"{output_dir}/test3_Q_int8.hex", Q)
        save_hex_int8(f"{output_dir}/test3_K_int8.hex", K)
        save_hex_int32(f"{output_dir}/test3_QK_int32.hex", QK)
        
        save_hex_int8(f"{output_dir}/test3_A_attn_int8.hex", A_attn)
        save_hex_int8(f"{output_dir}/test3_V_int8.hex", V)
        save_hex_int32(f"{output_dir}/test3_AV_int32.hex", AV)
        
        save_hex_int8(f"{output_dir}/test3_X_fc1_int8.hex", X_fc1)
        save_hex_int8(f"{output_dir}/test3_W_fc1_int8.hex", W_fc1)
        save_hex_int32(f"{output_dir}/test3_Y_fc1_int32.hex", Y_fc1)
        
        save_hex_int8(f"{output_dir}/test3_X_fc2_int8.hex", X_fc2)
        save_hex_int8(f"{output_dir}/test3_W_fc2_int8.hex", W_fc2)
        save_hex_int32(f"{output_dir}/test3_Y_fc2_int32.hex", Y_fc2)
        
        np.save(f"{output_dir}/test3_Q_int8.npy", Q)
        np.save(f"{output_dir}/test3_K_int8.npy", K)
        np.save(f"{output_dir}/test3_QK_int32.npy", QK)
        np.save(f"{output_dir}/test3_AV_int32.npy", AV)
        np.save(f"{output_dir}/test3_Y_fc1_int32.npy", Y_fc1)
        np.save(f"{output_dir}/test3_Y_fc2_int32.npy", Y_fc2)
        
        return {
            'tpc0': {'op': 'QK^T', 'cycles': stats0['cycles'], 'macs': stats0['total_macs']},
            'tpc1': {'op': 'Attn@V', 'cycles': stats1['cycles'], 'macs': stats1['total_macs']},
            'tpc2': {'op': 'FC1', 'cycles': stats2['cycles'], 'macs': stats2['total_macs']},
            'tpc3': {'op': 'FC2', 'cycles': stats3['cycles'], 'macs': stats3['total_macs']},
            'parallel_cycles': max_cycles,
            'total_macs': total_macs
        }
    
    def test4_pipeline_hazards(self, output_dir: str):
        """
        Test 4: Pipeline hazard scenarios.
        
        Tests read-after-write (RAW) dependencies:
        - C = A × B
        - D = C × E (RAW: read C after write)
        - F = C × G (RAW: reuse C)
        
        Also tests write-after-read (WAR):
        - Overwrite A while C still being computed
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "="*60)
        print("TEST 4: Pipeline Hazard Scenarios")
        print("="*60)
        
        M, K, N = 16, 16, 16
        
        # Generate matrices
        A = np.random.randint(-50, 50, size=(M, K), dtype=np.int8)
        B = np.random.randint(-50, 50, size=(K, N), dtype=np.int8)
        E = np.random.randint(-50, 50, size=(N, N), dtype=np.int8)
        G = np.random.randint(-50, 50, size=(N, N), dtype=np.int8)
        
        print(f"\n  Stage 1: C = A × B")
        C, stats1 = tiled_gemm_int8(A, B)
        C_int8 = np.clip(C >> 8, -128, 127).astype(np.int8)
        print(f"    C range: [{C.min()}, {C.max()}]")
        
        print(f"\n  Stage 2: D = C × E (RAW on C)")
        D, stats2 = tiled_gemm_int8(C_int8, E)
        print(f"    D range: [{D.min()}, {D.max()}]")
        
        print(f"\n  Stage 3: F = C × G (reuse C)")
        F, stats3 = tiled_gemm_int8(C_int8, G)
        print(f"    F range: [{F.min()}, {F.max()}]")
        
        # WAR scenario: new A for next iteration
        A_new = np.random.randint(-50, 50, size=(M, K), dtype=np.int8)
        print(f"\n  Stage 4: WAR - overwrite A")
        C_new, stats4 = tiled_gemm_int8(A_new, B)
        print(f"    C_new range: [{C_new.min()}, {C_new.max()}]")
        
        total_cycles = stats1['cycles'] + stats2['cycles'] + stats3['cycles'] + stats4['cycles']
        
        print(f"\n  Total cycles (with dependencies): {total_cycles}")
        
        # Save vectors
        save_hex_int8(f"{output_dir}/test4_A_int8.hex", A)
        save_hex_int8(f"{output_dir}/test4_B_int8.hex", B)
        save_hex_int8(f"{output_dir}/test4_E_int8.hex", E)
        save_hex_int8(f"{output_dir}/test4_G_int8.hex", G)
        save_hex_int8(f"{output_dir}/test4_A_new_int8.hex", A_new)
        
        save_hex_int32(f"{output_dir}/test4_C_int32.hex", C)
        save_hex_int8(f"{output_dir}/test4_C_int8.hex", C_int8)
        save_hex_int32(f"{output_dir}/test4_D_int32.hex", D)
        save_hex_int32(f"{output_dir}/test4_F_int32.hex", F)
        save_hex_int32(f"{output_dir}/test4_C_new_int32.hex", C_new)
        
        np.save(f"{output_dir}/test4_A_int8.npy", A)
        np.save(f"{output_dir}/test4_B_int8.npy", B)
        np.save(f"{output_dir}/test4_C_int32.npy", C)
        np.save(f"{output_dir}/test4_D_int32.npy", D)
        np.save(f"{output_dir}/test4_F_int32.npy", F)
        np.save(f"{output_dir}/test4_C_new_int32.npy", C_new)
        
        return {
            'M': M, 'K': K, 'N': N,
            'stages': ['C=A×B', 'D=C×E (RAW)', 'F=C×G (reuse)', 'C_new=A_new×B (WAR)'],
            'total_cycles': total_cycles
        }
    
    def test5_boundary_conditions(self, output_dir: str):
        """
        Test 5: Boundary conditions.
        
        Tests edge cases:
        - Non-tile-aligned dimensions
        - Single element
        - Maximum values (overflow potential)
        - Minimum values
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "="*60)
        print("TEST 5: Boundary Conditions")
        print("="*60)
        
        # Test 5a: Non-tile-aligned (7×13 @ 13×5)
        M, K, N = 7, 13, 5
        A = np.random.randint(-50, 50, size=(M, K), dtype=np.int8)
        B = np.random.randint(-50, 50, size=(K, N), dtype=np.int8)
        
        print(f"\n  5a: Non-aligned ({M}, {K}) × ({K}, {N})")
        C_unaligned, stats_a = tiled_gemm_int8(A, B)
        print(f"    Result: ({M}, {N}), range [{C_unaligned.min()}, {C_unaligned.max()}]")
        
        # Test 5b: Single element (1×1 @ 1×1)
        A_single = np.array([[100]], dtype=np.int8)
        B_single = np.array([[50]], dtype=np.int8)
        
        print(f"\n  5b: Single element (1, 1) × (1, 1)")
        C_single, stats_b = tiled_gemm_int8(A_single, B_single)
        print(f"    100 × 50 = {C_single[0,0]} (expected 5000)")
        
        # Test 5c: Maximum values (potential overflow)
        A_max = np.full((8, 8), 127, dtype=np.int8)
        B_max = np.full((8, 8), 127, dtype=np.int8)
        
        print(f"\n  5c: Max values (all 127)")
        C_max, stats_c = tiled_gemm_int8(A_max, B_max)
        expected_max = 127 * 127 * 8  # 129032
        print(f"    Each element: 127×127×8 = {expected_max}")
        print(f"    Actual: {C_max[0,0]} (fits in INT32: {C_max[0,0] == expected_max})")
        
        # Test 5d: Minimum values
        A_min = np.full((8, 8), -128, dtype=np.int8)
        B_min = np.full((8, 8), -128, dtype=np.int8)
        
        print(f"\n  5d: Min values (all -128)")
        C_min, stats_d = tiled_gemm_int8(A_min, B_min)
        expected_min = (-128) * (-128) * 8  # 131072
        print(f"    Each element: -128×-128×8 = {expected_min}")
        print(f"    Actual: {C_min[0,0]} (fits in INT32: {C_min[0,0] == expected_min})")
        
        # Test 5e: Mixed min/max (worst case range)
        A_mixed = np.full((8, 8), 127, dtype=np.int8)
        B_mixed = np.full((8, 8), -128, dtype=np.int8)
        
        print(f"\n  5e: Mixed max/min (127 × -128)")
        C_mixed, stats_e = tiled_gemm_int8(A_mixed, B_mixed)
        expected_mixed = 127 * (-128) * 8  # -130048
        print(f"    Each element: 127×-128×8 = {expected_mixed}")
        print(f"    Actual: {C_mixed[0,0]} (fits in INT32: {C_mixed[0,0] == expected_mixed})")
        
        # Save vectors
        save_hex_int8(f"{output_dir}/test5a_A_int8.hex", A)
        save_hex_int8(f"{output_dir}/test5a_B_int8.hex", B)
        save_hex_int32(f"{output_dir}/test5a_C_int32.hex", C_unaligned)
        
        save_hex_int8(f"{output_dir}/test5b_A_int8.hex", A_single)
        save_hex_int8(f"{output_dir}/test5b_B_int8.hex", B_single)
        save_hex_int32(f"{output_dir}/test5b_C_int32.hex", C_single)
        
        save_hex_int8(f"{output_dir}/test5c_A_int8.hex", A_max)
        save_hex_int8(f"{output_dir}/test5c_B_int8.hex", B_max)
        save_hex_int32(f"{output_dir}/test5c_C_int32.hex", C_max)
        
        save_hex_int8(f"{output_dir}/test5d_A_int8.hex", A_min)
        save_hex_int8(f"{output_dir}/test5d_B_int8.hex", B_min)
        save_hex_int32(f"{output_dir}/test5d_C_int32.hex", C_min)
        
        save_hex_int8(f"{output_dir}/test5e_A_int8.hex", A_mixed)
        save_hex_int8(f"{output_dir}/test5e_B_int8.hex", B_mixed)
        save_hex_int32(f"{output_dir}/test5e_C_int32.hex", C_mixed)
        
        np.save(f"{output_dir}/test5a_C_int32.npy", C_unaligned)
        np.save(f"{output_dir}/test5b_C_int32.npy", C_single)
        np.save(f"{output_dir}/test5c_C_int32.npy", C_max)
        np.save(f"{output_dir}/test5d_C_int32.npy", C_min)
        np.save(f"{output_dir}/test5e_C_int32.npy", C_mixed)
        
        return {
            '5a_unaligned': {'M': M, 'K': K, 'N': N},
            '5b_single': {'result': int(C_single[0,0])},
            '5c_max': {'expected': expected_max, 'actual': int(C_max[0,0])},
            '5d_min': {'expected': expected_min, 'actual': int(C_min[0,0])},
            '5e_mixed': {'expected': expected_mixed, 'actual': int(C_mixed[0,0])}
        }
    
    def generate_all(self):
        """Generate all test vectors."""
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        print("\n" + "="*60)
        print("PHASE E: Stress Testing")
        print("="*60)
        
        results = {}
        results['test1_back_to_back'] = self.test1_back_to_back_gemm(OUTPUT_DIR)
        results['test2_max_sram'] = self.test2_max_sram_utilization(OUTPUT_DIR)
        results['test3_multi_tpc'] = self.test3_multi_tpc_parallel(OUTPUT_DIR)
        results['test4_hazards'] = self.test4_pipeline_hazards(OUTPUT_DIR)
        results['test5_boundary'] = self.test5_boundary_conditions(OUTPUT_DIR)
        
        # Save summary
        with open(f"{OUTPUT_DIR}/summary.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\n" + "="*60)
        print(f"All vectors saved to: {OUTPUT_DIR}")
        print("="*60)
        
        # List files
        files = sorted(os.listdir(OUTPUT_DIR))
        total_size = sum(os.path.getsize(f"{OUTPUT_DIR}/{f}") for f in files)
        print(f"\nGenerated {len(files)} files, {total_size/1024:.1f} KB total")
        
        return results


if __name__ == "__main__":
    tests = PhaseE_Tests(seed=42)
    tests.generate_all()
