
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001600 <E_2>:
	s_load_b128 s[4:7], s[0:1], null                           // 000000001604: F4080100 F8000000
	v_mov_b32_e32 v0, 0                                        // 000000001614: 7E000280
	s_load_b32 s2, s[6:7], 0x4                                 // 00000000161C: F4000083 F8000004
	v_add_f32_e64 v1, s2, s2                                   // 000000001630: D5030001 00000002
	global_store_b32 v0, v1, s[4:5] offset:4                   // 000000001638: DC6A0004 00040100
	s_endpgm                                                   // 000000001648: BFB00000
