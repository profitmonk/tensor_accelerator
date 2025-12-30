#!/usr/bin/env python3
"""
NoC (Network-on-Chip) Cycle-Accurate Model

Implements a 2D mesh network with XY routing:
- 5-port routers (N, S, E, W, Local)
- X-first then Y routing (deadlock-free)
- FIFO buffering at inputs
- Round-robin arbitration

Author: Tensor Accelerator Project
"""

from collections import deque
from enum import IntEnum
from dataclasses import dataclass
from typing import Optional, List, Tuple
import numpy as np


class Port(IntEnum):
    """Router port indices"""
    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3
    LOCAL = 4


@dataclass
class Packet:
    """NoC packet structure"""
    data: int              # 256-bit payload
    dest_x: int           # Destination X coordinate
    dest_y: int           # Destination Y coordinate
    src_x: int = 0        # Source X (for debugging)
    src_y: int = 0        # Source Y (for debugging)


class InputBuffer:
    """FIFO input buffer for a router port"""
    
    def __init__(self, depth: int = 4):
        self.depth = depth
        self.fifo = deque(maxlen=depth)
    
    def push(self, packet: Packet) -> bool:
        """Push packet, return True if successful"""
        if len(self.fifo) < self.depth:
            self.fifo.append(packet)
            return True
        return False
    
    def pop(self) -> Optional[Packet]:
        """Pop and return front packet"""
        if self.fifo:
            return self.fifo.popleft()
        return None
    
    def peek(self) -> Optional[Packet]:
        """Peek at front packet without removing"""
        if self.fifo:
            return self.fifo[0]
        return None
    
    def is_empty(self) -> bool:
        return len(self.fifo) == 0
    
    def is_full(self) -> bool:
        return len(self.fifo) >= self.depth
    
    @property
    def ready(self) -> bool:
        """Can accept new packet"""
        return not self.is_full()


class NoCRouter:
    """
    5-port NoC router with XY routing
    
    Ports: NORTH, SOUTH, EAST, WEST, LOCAL
    Routing: X dimension first, then Y (deadlock-free)
    """
    
    def __init__(self, x: int, y: int, fifo_depth: int = 4, verbose: bool = False):
        self.x = x
        self.y = y
        self.verbose = verbose
        
        # Input buffers for each port
        self.input_buffers = [InputBuffer(fifo_depth) for _ in range(5)]
        
        # Output holding registers (pending packets to send)
        self.output_pending = [None for _ in range(5)]  # Packet waiting to be sent
        self.output_ready = [True for _ in range(5)]    # Downstream ready signals
        
        # Round-robin arbitration state
        self.arb_priority = 0
        
        # Statistics
        self.packets_routed = 0
        self.packets_received = 0  # Delivered to local
    
    def compute_output_port(self, packet: Packet) -> int:
        """XY routing: X first, then Y"""
        # X routing first
        if packet.dest_x > self.x:
            return Port.EAST
        elif packet.dest_x < self.x:
            return Port.WEST
        # X matches, now Y routing
        elif packet.dest_y > self.y:
            return Port.NORTH
        elif packet.dest_y < self.y:
            return Port.SOUTH
        else:
            # Destination is this router
            return Port.LOCAL
    
    def input_ready(self, port: int) -> bool:
        """Check if input port can accept packet"""
        return self.input_buffers[port].ready
    
    def inject(self, port: int, packet: Packet) -> bool:
        """Inject packet into input buffer"""
        if self.input_buffers[port].push(packet):
            if self.verbose:
                print(f"  [Router ({self.x},{self.y})] Injected packet at port {Port(port).name} -> dest ({packet.dest_x},{packet.dest_y})")
            return True
        return False
    
    def set_output_ready(self, port: int, ready: bool):
        """Set downstream ready signal for output port"""
        self.output_ready[port] = ready
    
    def get_output(self, port: int) -> Optional[Packet]:
        """Get packet from output port (if any)"""
        return self.output_pending[port]
    
    def clear_output(self, port: int):
        """Clear output after downstream accepts"""
        self.output_pending[port] = None
    
    def posedge(self):
        """Process one clock cycle"""
        # Phase 1: Move pending outputs that were accepted
        for port in range(5):
            if self.output_pending[port] and self.output_ready[port]:
                if self.verbose:
                    pkt = self.output_pending[port]
                    print(f"  [Router ({self.x},{self.y})] Sent packet via {Port(port).name}")
                self.output_pending[port] = None
        
        # Phase 2: Route new packets from input buffers
        # Round-robin arbitration across all input ports
        for i in range(5):
            in_port = (self.arb_priority + i) % 5
            buf = self.input_buffers[in_port]
            
            if buf.is_empty():
                continue
            
            packet = buf.peek()
            out_port = self.compute_output_port(packet)
            
            # Check if output port is free
            if self.output_pending[out_port] is None:
                # Route the packet
                buf.pop()
                self.output_pending[out_port] = packet
                self.packets_routed += 1
                
                if out_port == Port.LOCAL:
                    self.packets_received += 1
                
                if self.verbose:
                    print(f"  [Router ({self.x},{self.y})] Routed {Port(in_port).name} -> {Port(out_port).name}")
                
                # Update round-robin priority
                self.arb_priority = (in_port + 1) % 5
                break
        
        # Advance arbitration even if no packet routed
        self.arb_priority = (self.arb_priority + 1) % 5


class NoCMesh:
    """
    2D Mesh NoC
    
    Connects routers in a grid topology.
    """
    
    def __init__(self, width: int, height: int, fifo_depth: int = 4, verbose: bool = False):
        self.width = width
        self.height = height
        self.verbose = verbose
        self.cycle = 0
        
        # Create router grid
        self.routers = [[NoCRouter(x, y, fifo_depth, verbose) 
                        for y in range(height)] 
                       for x in range(width)]
        
        # Local receive buffers (for TPC interface)
        self.local_rx = [[deque() for y in range(height)] for x in range(width)]
    
    def get_router(self, x: int, y: int) -> NoCRouter:
        return self.routers[x][y]
    
    def inject(self, src_x: int, src_y: int, data: int, dest_x: int, dest_y: int) -> bool:
        """Inject packet from TPC at (src_x, src_y) to (dest_x, dest_y)"""
        packet = Packet(data=data, dest_x=dest_x, dest_y=dest_y, src_x=src_x, src_y=src_y)
        router = self.routers[src_x][src_y]
        return router.inject(Port.LOCAL, packet)
    
    def receive(self, x: int, y: int) -> Optional[Packet]:
        """Receive packet at TPC (x, y) - returns None if no packet"""
        if self.local_rx[x][y]:
            return self.local_rx[x][y].popleft()
        return None
    
    def posedge(self):
        """Advance one clock cycle"""
        self.cycle += 1
        
        # Phase 1: Process all routers
        for x in range(self.width):
            for y in range(self.height):
                self.routers[x][y].posedge()
        
        # Phase 2: Transfer packets between routers
        for x in range(self.width):
            for y in range(self.height):
                router = self.routers[x][y]
                
                # North output -> neighbor's South input
                if y < self.height - 1:
                    pkt = router.get_output(Port.NORTH)
                    if pkt:
                        neighbor = self.routers[x][y + 1]
                        if neighbor.inject(Port.SOUTH, pkt):
                            router.clear_output(Port.NORTH)
                
                # South output -> neighbor's North input
                if y > 0:
                    pkt = router.get_output(Port.SOUTH)
                    if pkt:
                        neighbor = self.routers[x][y - 1]
                        if neighbor.inject(Port.NORTH, pkt):
                            router.clear_output(Port.SOUTH)
                
                # East output -> neighbor's West input
                if x < self.width - 1:
                    pkt = router.get_output(Port.EAST)
                    if pkt:
                        neighbor = self.routers[x + 1][y]
                        if neighbor.inject(Port.WEST, pkt):
                            router.clear_output(Port.EAST)
                
                # West output -> neighbor's East input
                if x > 0:
                    pkt = router.get_output(Port.WEST)
                    if pkt:
                        neighbor = self.routers[x - 1][y]
                        if neighbor.inject(Port.EAST, pkt):
                            router.clear_output(Port.WEST)
                
                # Local output -> local receive buffer
                pkt = router.get_output(Port.LOCAL)
                if pkt:
                    self.local_rx[x][y].append(pkt)
                    router.clear_output(Port.LOCAL)


# =============================================================================
# Tests
# =============================================================================

def test_single_hop():
    """Test 1: Single hop East (0,0) -> (1,0)"""
    print("\n" + "="*60)
    print("TEST 1: Single hop East (0,0) → (1,0)")
    print("="*60)
    
    mesh = NoCMesh(2, 2, verbose=True)
    
    # Inject packet
    assert mesh.inject(0, 0, 0xDEADBEEF, 1, 0), "Injection failed"
    
    # Run until delivered
    for _ in range(10):
        mesh.posedge()
        pkt = mesh.receive(1, 0)
        if pkt:
            assert pkt.data == 0xDEADBEEF, f"Data mismatch: {pkt.data}"
            print(f"  Received at (1,0): data=0x{pkt.data:X}")
            print(">>> TEST 1 PASSED <<<")
            return True
    
    print("  FAIL: Packet not received")
    return False


def test_diagonal():
    """Test 2: Diagonal XY routing (0,0) -> (1,1)"""
    print("\n" + "="*60)
    print("TEST 2: Diagonal XY routing (0,0) → (1,1)")
    print("="*60)
    
    mesh = NoCMesh(2, 2, verbose=True)
    
    mesh.inject(0, 0, 0xCAFEBABE, 1, 1)
    
    for _ in range(20):
        mesh.posedge()
        pkt = mesh.receive(1, 1)
        if pkt:
            assert pkt.data == 0xCAFEBABE
            print(f"  Received at (1,1): data=0x{pkt.data:X}")
            print(">>> TEST 2 PASSED <<<")
            return True
    
    print("  FAIL: Packet not received")
    return False


def test_opposite_diagonal():
    """Test 3: Opposite diagonal (1,1) -> (0,0)"""
    print("\n" + "="*60)
    print("TEST 3: Opposite diagonal (1,1) → (0,0)")
    print("="*60)
    
    mesh = NoCMesh(2, 2, verbose=True)
    
    mesh.inject(1, 1, 0x12345678, 0, 0)
    
    for _ in range(20):
        mesh.posedge()
        pkt = mesh.receive(0, 0)
        if pkt:
            assert pkt.data == 0x12345678
            print(f"  Received at (0,0): data=0x{pkt.data:X}")
            print(">>> TEST 3 PASSED <<<")
            return True
    
    print("  FAIL: Packet not received")
    return False


def test_self_delivery():
    """Test 4: Self delivery (1,0) -> (1,0)"""
    print("\n" + "="*60)
    print("TEST 4: Self delivery (1,0) → (1,0)")
    print("="*60)
    
    mesh = NoCMesh(2, 2, verbose=True)
    
    mesh.inject(1, 0, 0xAAAABBBB, 1, 0)
    
    for _ in range(10):
        mesh.posedge()
        pkt = mesh.receive(1, 0)
        if pkt:
            assert pkt.data == 0xAAAABBBB
            print(f"  Received at (1,0): data=0x{pkt.data:X}")
            print(">>> TEST 4 PASSED <<<")
            return True
    
    print("  FAIL: Packet not received")
    return False


def test_multiple_packets():
    """Test 5: Multiple concurrent packets"""
    print("\n" + "="*60)
    print("TEST 5: Multiple concurrent packets (all corners)")
    print("="*60)
    
    mesh = NoCMesh(2, 2, verbose=False)
    
    # Inject from all 4 corners to opposite corners
    mesh.inject(0, 0, 0x00000001, 1, 1)
    mesh.inject(1, 1, 0x00000002, 0, 0)
    mesh.inject(0, 1, 0x00000003, 1, 0)
    mesh.inject(1, 0, 0x00000004, 0, 1)
    
    received = {(0,0): None, (1,0): None, (0,1): None, (1,1): None}
    
    for _ in range(50):
        mesh.posedge()
        for x in range(2):
            for y in range(2):
                pkt = mesh.receive(x, y)
                if pkt:
                    received[(x, y)] = pkt.data
    
    print(f"  (0,0) received: {received[(0,0)]}")
    print(f"  (1,0) received: {received[(1,0)]}")
    print(f"  (0,1) received: {received[(0,1)]}")
    print(f"  (1,1) received: {received[(1,1)]}")
    
    assert received[(1,1)] == 0x00000001, "Packet to (1,1) failed"
    assert received[(0,0)] == 0x00000002, "Packet to (0,0) failed"
    assert received[(1,0)] == 0x00000003, "Packet to (1,0) failed"
    assert received[(0,1)] == 0x00000004, "Packet to (0,1) failed"
    
    print(">>> TEST 5 PASSED <<<")
    return True


def main():
    print("="*60)
    print("NoC MESH MODEL TESTS")
    print("="*60)
    
    tests = [
        test_single_hop,
        test_diagonal,
        test_opposite_diagonal,
        test_self_delivery,
        test_multiple_packets,
    ]
    
    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"  EXCEPTION: {e}")
    
    print("\n" + "="*60)
    print(f"NOC MODEL: {passed}/{len(tests)} tests passed")
    print("="*60)
    
    if passed == len(tests):
        print(">>> ALL NOC MODEL TESTS PASSED! <<<")
        return 0
    return 1


if __name__ == "__main__":
    exit(main())
