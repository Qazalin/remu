
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001600 <E_10n42>:
	s_load_b128 s[0:3], s[0:1], null                           // 000000001600: F4080000 F8000000
	v_mov_b32_e32 v0, 0                                        // 000000001608: 7E000280
	s_ashr_i32 s4, s15, 31                                     // 00000000160C: 86049F0F
	s_waitcnt lgkmcnt(0)                                       // 000000001610: BF89FC07
	s_add_u32 s2, s2, s15                                      // 000000001614: 80020F02
	s_addc_u32 s3, s3, s4                                      // 000000001618: 82030403
	s_add_u32 s0, s0, s15                                      // 00000000161C: 80000F00
	global_load_u8 v1, v0, s[2:3]                              // 000000001620: DC420000 01020000
	s_addc_u32 s1, s1, s4                                      // 000000001628: 82010401
	s_waitcnt vmcnt(0)                                         // 00000000162C: BF8903F7
	global_store_b8 v0, v1, s[0:1]                             // 000000001630: DC620000 00000100
	s_nop 0                                                    // 000000001638: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 00000000163C: BFB60003
	s_endpgm                                                   // 000000001640: BFB00000
