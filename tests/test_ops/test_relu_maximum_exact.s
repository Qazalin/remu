
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001600 <E_3n1>:
	s_load_b128 s[0:3], s[0:1], null                           // 000000001600: F4080000 F8000000
	s_mov_b32 s4, s15                                          // 000000001608: BE84000F
	s_ashr_i32 s5, s15, 31                                     // 00000000160C: 86059F0F
	v_mov_b32_e32 v1, 0                                        // 000000001610: 7E020280
	s_lshl_b64 s[4:5], s[4:5], 2                               // 000000001614: 84848204
	s_waitcnt lgkmcnt(0)                                       // 000000001618: BF89FC07
	s_add_u32 s2, s2, s4                                       // 00000000161C: 80020402
	s_addc_u32 s3, s3, s5                                      // 000000001620: 82030503
	s_add_u32 s0, s0, s4                                       // 000000001624: 80000400
	s_load_b32 s2, s[2:3], null                                // 000000001628: F4000081 F8000000
	s_addc_u32 s1, s1, s5                                      // 000000001630: 82010501
	s_waitcnt lgkmcnt(0)                                       // 000000001634: BF89FC07
	v_mul_f32_e64 v0, s2, 0.5                                  // 000000001638: D5080000 0001E002
	v_cmp_gt_f32_e64 s3, s2, 0                                 // 000000001640: D4140003 00010002
	v_cmp_nlt_f32_e64 vcc_lo, s2, 0                            // 000000001648: D41E006A 00010002
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001650: BF870092
	v_cndmask_b32_e64 v0, v0, s2, s3                           // 000000001654: D5010000 000C0500
	v_cndmask_b32_e32 v0, 0, v0, vcc_lo                        // 00000000165C: 02000080
	global_store_b32 v1, v0, s[0:1]                            // 000000001660: DC6A0000 00000001
	s_nop 0                                                    // 000000001668: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 00000000166C: BFB60003
	s_endpgm                                                   // 000000001670: BFB00000
