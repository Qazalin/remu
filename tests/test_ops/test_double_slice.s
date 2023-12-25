
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001600 <E_n3>:
	s_load_b128 s[0:3], s[0:1], null                           // 000000001600: F4080000 F8000000
	s_waitcnt lgkmcnt(0)                                       // 000000001608: BF89FC07
	s_load_b32 s2, s[2:3], 0x14                                // 00000000160C: F4000081 F8000014
	s_waitcnt lgkmcnt(0)                                       // 000000001614: BF89FC07
	v_dual_mov_b32 v0, 0 :: v_dual_mov_b32 v1, s2              // 000000001618: CA100080 00000002
	global_store_b32 v0, v1, s[0:1]                            // 000000001620: DC6A0000 00000100
	s_nop 0                                                    // 000000001628: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 00000000162C: BFB60003
	s_endpgm                                                   // 000000001630: BFB00000
