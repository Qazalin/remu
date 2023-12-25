
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001600 <r_256_256n2>:
	s_load_b64 s[0:1], s[0:1], null                            // 000000001600: F4040000 F8000000
	s_mov_b32 s2, s15                                          // 000000001608: BE82000F
	s_ashr_i32 s3, s15, 31                                     // 00000000160C: 86039F0F
	v_dual_mov_b32 v0, 0 :: v_dual_mov_b32 v1, 0x43800000      // 000000001610: CA100080 000000FF 43800000
	s_lshl_b64 s[2:3], s[2:3], 2                               // 00000000161C: 84828202
	s_waitcnt lgkmcnt(0)                                       // 000000001620: BF89FC07
	s_add_u32 s0, s0, s2                                       // 000000001624: 80000200
	s_addc_u32 s1, s1, s3                                      // 000000001628: 82010301
	global_store_b32 v0, v1, s[0:1]                            // 00000000162C: DC6A0000 00000100
	s_nop 0                                                    // 000000001634: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000001638: BFB60003
	s_endpgm                                                   // 00000000163C: BFB00000
