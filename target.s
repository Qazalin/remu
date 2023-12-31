
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001600 <E_2>:
	s_load_b128 s[0:3], s[0:1], null                           // 000000001600: F4080000 F8000000
	v_mov_b32_e32 v0, 0                                        // 000000001608: 7E000280
	s_waitcnt lgkmcnt(0)                                       // 00000000160C: BF89FC07
	s_load_b32 s2, s[2:3], 0x4                                 // 000000001610: F4000081 F8000004
	s_waitcnt lgkmcnt(0)                                       // 000000001618: BF89FC07
	v_add_f32_e64 v1, s2, s2                                   // 00000000161C: D5030001 00000402
	global_store_b32 v0, v1, s[0:1] offset:4                   // 000000001624: DC6A0004 00000100
	s_nop 0                                                    // 00000000162C: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000001630: BFB60003
	s_endpgm                                                   // 000000001634: BFB00000

tinygrad:  [-0.372549  4.      ]
