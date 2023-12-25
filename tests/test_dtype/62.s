
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001600 <E_4n22>:
	s_load_b128 s[0:3], s[0:1], null                           // 000000001600: F4080000 F8000000
	v_mov_b32_e32 v0, 0                                        // 000000001608: 7E000280
	s_ashr_i32 s5, s15, 31                                     // 00000000160C: 86059F0F
	s_mov_b32 s4, s15                                          // 000000001610: BE84000F
	s_waitcnt lgkmcnt(0)                                       // 000000001614: BF89FC07
	s_add_u32 s2, s2, s15                                      // 000000001618: 80020F02
	s_addc_u32 s3, s3, s5                                      // 00000000161C: 82030503
	global_load_u8 v1, v0, s[2:3]                              // 000000001620: DC420000 01020000
	s_lshl_b64 s[2:3], s[4:5], 1                               // 000000001628: 84828104
	s_delay_alu instid0(SALU_CYCLE_1)                          // 00000000162C: BF870009
	s_add_u32 s0, s0, s2                                       // 000000001630: 80000200
	s_addc_u32 s1, s1, s3                                      // 000000001634: 82010301
	s_waitcnt vmcnt(0)                                         // 000000001638: BF8903F7
	v_cvt_f16_u16_e32 v1, v1                                   // 00000000163C: 7E02A101
	global_store_b16 v0, v1, s[0:1]                            // 000000001640: DC660000 00000100
	s_nop 0                                                    // 000000001648: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 00000000164C: BFB60003
	s_endpgm                                                   // 000000001650: BFB00000
